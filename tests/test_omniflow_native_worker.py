"""Native OmegaNode SDK acceptance: no ComfyUI, neural weights or GPU required."""
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "omniflow-worker"))
from omega_sdk.protocol import (
    OMEGA_ABI_VERSION, ArtifactHandle, ContentAddressedStorage,
    NodeExecutionMode, NodeManifest, OmegaNode, OmegaWorkerRpcServer, TrustClass,
)


class EchoNode(OmegaNode):
    @classmethod
    def manifest(cls):
        return NodeManifest("T001_Echo", "1.1.0", "TEST", OMEGA_ABI_VERSION,
                            NodeExecutionMode.PURE_DETERMINISTIC,
                            TrustClass.T0_CORE_TRUSTED, (), False)

    def execute(self, ctx, inputs, config):
        return {"echo": inputs["value"], "tenant": ctx.tenant_id}


class UntrustedEcho(EchoNode):
    @classmethod
    def manifest(cls):
        return NodeManifest("T002_Untrusted", "1.1.0", "TEST", OMEGA_ABI_VERSION,
                            NodeExecutionMode.PURE_DETERMINISTIC,
                            TrustClass.T2_UNTRUSTED_OMEGANODE, (), False)


class ExternalEcho(EchoNode):
    @classmethod
    def manifest(cls):
        return NodeManifest("T003_External", "1.1.0", "TEST", OMEGA_ABI_VERSION,
                            NodeExecutionMode.EXTERNAL_SIDE_EFFECT,
                            TrustClass.T0_CORE_TRUSTED, (), False)


class NativeOmegaNodeAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.cas = ContentAddressedStorage(root / "cas")
        self.worker = OmegaWorkerRpcServer(root / "worker.sock", self.cas)
        self.worker.register_operator(EchoNode())

    def request(self, **updates):
        payload = {
            "abi_version": OMEGA_ABI_VERSION,
            "op_id": "T001_Echo", "op_version": "1.1.0",
            "job_id": "job_001", "tenant_id": "tenant_A",
            "fencing_token": 1, "fidelity_mode": "DOC",
            "seed": 123, "inputs": {"value": 42}, "config": {},
        }
        payload.update(updates)
        return json.loads(self.worker.handle_frame(json.dumps(payload).encode()))

    def test_valid_operator_executes_and_returns_fencing_token(self):
        result = self.request()
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["fencing_token"], 1)
        self.assertEqual(result["outputs"], {"echo": 42, "tenant": "tenant_A"})
        self.assertEqual(result["abi_version"], OMEGA_ABI_VERSION)

    def test_missing_operator_is_rejected(self):
        self.assertEqual(self.request(op_id="missing")["code"], "ERR_OPERATOR_NOT_REGISTERED")

    def test_wrong_abi_is_rejected(self):
        self.assertEqual(self.request(abi_version="unknown.v9")["code"], "ERR_ABI_MISMATCH")

    def test_t2_cannot_execute_without_sandbox(self):
        self.worker.register_operator(UntrustedEcho())
        self.assertEqual(self.request(op_id="T002_Untrusted")["code"], "ERR_ISOLATION_NOT_READY")

    def test_side_effect_operator_requires_control_plane(self):
        self.worker.register_operator(ExternalEcho())
        self.assertEqual(self.request(op_id="T003_External")["code"], "ERR_APPROVAL_REQUIRED")

    def test_tenant_path_traversal_is_blocked(self):
        self.assertEqual(self.request(tenant_id="../outside")["status"], "ERROR")

    def test_missing_or_zero_fencing_is_rejected(self):
        self.assertEqual(self.request(fencing_token=0)["status"], "ERROR")

    def test_unsafe_inputs_fail_closed(self):
        self.assertEqual(self.request(inputs=[])["status"], "ERROR")
        self.assertEqual(json.loads(self.worker.handle_frame(b"not-json"))["status"], "ERROR")
        self.assertEqual(json.loads(self.worker.handle_frame(b"[]"))["status"], "ERROR")

    def test_duplicate_operator_rejected(self):
        with self.assertRaises(ValueError):
            self.worker.register_operator(EchoNode())

    def test_cas_commits_immutable_source_with_sha256(self):
        path = self.cas.tmp_dir / "input.partial"
        path.write_bytes(b"camera-original")
        result = self.cas.commit_file_atomically(path, "tenant_A", "image/tiff", "ICC16")
        self.assertIsInstance(result, ArtifactHandle)
        self.assertEqual(result.sha256, "sha256:" + hashlib.sha256(b"camera-original").hexdigest())
        self.assertEqual(Path(result.cas_path).read_bytes(), b"camera-original")
        self.assertFalse(path.exists())

    def test_cas_duplicate_content_is_idempotent(self):
        path = self.cas.tmp_dir / "first.partial"
        path.write_bytes(b"same")
        first = self.cas.commit_file_atomically(path, "tenant_A", "x/test", "RGB")
        duplicate = self.cas.tmp_dir / "second.partial"
        duplicate.write_bytes(b"same")
        second = self.cas.commit_file_atomically(duplicate, "tenant_A", "x/test", "RGB")
        self.assertEqual(first.cas_path, second.cas_path)

    def test_cas_rejects_external_path_and_cross_tenant_traversal(self):
        path = Path(self.temp.name) / "raw.nef"
        path.write_bytes(b"original")
        with self.assertRaises(ValueError):
            self.cas.commit_file_atomically(path, "tenant_A", "image/x-nikon-nef", "RAW")
        inside = self.cas.tmp_dir / "raw.nef"
        inside.write_bytes(b"original")
        with self.assertRaises(ValueError):
            self.cas.commit_file_atomically(inside, "../tenant_B", "image/x-nikon-nef", "RAW")

    def test_giant_rpc_frame_is_rejected_before_allocation(self):
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        client.sendall((10_000_000).to_bytes(4, "big"))
        self.worker.handle_connection(server)
        length = int.from_bytes(client.recv(4), "big")
        self.assertEqual(json.loads(client.recv(length))["code"], "ERR_FRAME_TOO_LARGE")


if __name__ == "__main__":
    unittest.main()
