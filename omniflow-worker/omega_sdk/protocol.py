"""Native AETHERIS OmniFlow OmegaNode SDK v1.1; SYS-AETHERIS-OMNIFLOW-016.

T0-only baseline. Isolation of T1/T2 plugins, authoritative fencing,
identity and authentication are deliberately NOT claimed until implemented.
No ComfyUI runtime dependency.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import struct
from typing import Any, Generic, TypeVar

OMEGA_ABI_VERSION = "omega.node.abi.v1.1"
MAX_RPC_FRAME_BYTES = 1 << 20
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class NodeExecutionMode(str, Enum):
    PURE_DETERMINISTIC = "PURE_DETERMINISTIC"
    PURE_SEEDED = "PURE_SEEDED"
    IMPURE_IDEMPOTENT = "IMPURE_IDEMPOTENT"
    EXTERNAL_SIDE_EFFECT = "EXTERNAL_SIDE_EFFECT"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"


class TrustClass(str, Enum):
    T0_CORE_TRUSTED = "T0_CORE_TRUSTED"
    T1_VERIFIED_PLUGIN = "T1_VERIFIED_PLUGIN"
    T2_UNTRUSTED_OMEGANODE = "T2_UNTRUSTED_OMEGANODE"


@dataclass(frozen=True)
class ArtifactHandle:
    sha256: str
    cas_path: str
    mime_type: str
    byte_size: int
    tenant_id: str
    color_space: str
    icc_profile_sha256: str | None = None


@dataclass(frozen=True)
class NodeManifest:
    op_id: str
    op_version: str
    family: str
    abi_version: str
    mode: NodeExecutionMode
    trust_class: TrustClass
    required_model_ids: tuple[str, ...]
    supports_tiling: bool


@dataclass(frozen=True)
class ExecutionContext:
    job_id: str
    tenant_id: str
    fencing_token: int
    fidelity_mode: str
    seed: int
    cas_root: Path
    scratch_dir: Path


TInput = TypeVar("TInput", bound=dict[str, Any])
TOutput = TypeVar("TOutput", bound=dict[str, Any])


class OmegaNode(ABC, Generic[TInput, TOutput]):
    @classmethod
    @abstractmethod
    def manifest(cls) -> NodeManifest: ...

    @abstractmethod
    def execute(self, ctx: ExecutionContext, inputs: TInput,
                config: dict[str, Any]) -> TOutput: ...


def _validated_id(raw: Any, field: str) -> str:
    if not isinstance(raw, str) or _ID.fullmatch(raw) is None:
        raise ValueError(f"Invalid {field}")
    return raw


class ContentAddressedStorage:
    """Tenant-scoped immutable CAS; staged file must be on the same filesystem."""
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir).resolve()
        self.blobs_dir = self.root_dir / "blobs"
        self.tmp_dir = self.root_dir / "tmp"
        self.blobs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.tmp_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def commit_file_atomically(self, temp_file_path: Path, tenant_id: str,
                               mime_type: str, color_space: str,
                               icc_profile_sha256: str | None = None) -> ArtifactHandle:
        tenant_id = _validated_id(tenant_id, "tenant_id")
        candidate = Path(temp_file_path)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("Existing non-symlink staging file required")
        source = candidate.resolve(strict=True)
        if not source.is_relative_to(self.tmp_dir.resolve()):
            raise ValueError("Source must be inside CAS staging area")
        if not mime_type or not color_space:
            raise ValueError("MIME and color space required")
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                size += len(chunk)
                digest.update(chunk)
            os.fsync(handle.fileno())
        identity = digest.hexdigest()
        tenant_dir = self.blobs_dir / tenant_id / identity[:2]
        tenant_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        committed = tenant_dir / (identity + ".blob")
        if committed.exists():
            if committed.is_symlink() or not committed.is_file():
                raise OSError("Corrupt CAS target")
            if hashlib.sha256(committed.read_bytes()).hexdigest() != identity:
                raise OSError("CAS digest mismatch")
            source.unlink()
        else:
            # Exclusive hard link prevents replacement of committed artifacts.
            try:
                os.link(source, committed, follow_symlinks=False)
            except FileExistsError:
                if hashlib.sha256(committed.read_bytes()).hexdigest() != identity:
                    raise OSError("CAS concurrent digest mismatch")
            source.unlink()
            directory_fd = os.open(tenant_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return ArtifactHandle("sha256:" + identity, str(committed), mime_type,
                              size, tenant_id, color_space, icc_profile_sha256)


class _RpcRejection(ValueError):
    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code = code


class OmegaWorkerRpcServer:
    """Conservative T0-only local worker, NOT an auth or sandbox service."""
    def __init__(self, socket_path: Path, cas: ContentAddressedStorage):
        self.socket_path = Path(socket_path)
        self.cas = cas
        self._registry: dict[tuple[str, str], OmegaNode] = {}

    def register_operator(self, node: OmegaNode) -> None:
        manifest = node.manifest()
        if manifest.abi_version != OMEGA_ABI_VERSION:
            raise ValueError("Incompatible OmegaNode ABI")
        key = (manifest.op_id, manifest.op_version)
        if key in self._registry:
            raise ValueError("Duplicate OmegaNode operator")
        self._registry[key] = node

    @staticmethod
    def _response(payload: dict[str, Any]) -> bytes:
        return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")

    def handle_frame(self, payload: bytes) -> bytes:
        try:
            if len(payload) > MAX_RPC_FRAME_BYTES:
                raise _RpcRejection("ERR_FRAME_TOO_LARGE", "RPC frame exceeded limit")
            try:
                request = json.loads(payload)
            except (ValueError, UnicodeDecodeError) as exc:
                raise _RpcRejection("ERR_INVALID_FRAME", "Invalid JSON") from exc
            if not isinstance(request, dict):
                raise _RpcRejection("ERR_INVALID_FRAME", "RPC payload must be object")
            if request.get("abi_version") != OMEGA_ABI_VERSION:
                raise _RpcRejection("ERR_ABI_MISMATCH", "Incorrect ABI version")
            key = (request.get("op_id"), request.get("op_version"))
            try:
                node = self._registry.get(key)
            except TypeError as exc:
                raise _RpcRejection("ERR_INVALID_FRAME", "Malformed operator") from exc
            if node is None:
                raise _RpcRejection("ERR_OPERATOR_NOT_REGISTERED", "Unknown operator")
            manifest = node.manifest()
            if manifest.trust_class != TrustClass.T0_CORE_TRUSTED:
                raise _RpcRejection("ERR_ISOLATION_NOT_READY", "T1/T2 require isolation")
            if manifest.mode in (NodeExecutionMode.EXTERNAL_SIDE_EFFECT,
                                 NodeExecutionMode.HUMAN_APPROVAL):
                raise _RpcRejection("ERR_APPROVAL_REQUIRED", "Control plane approval required")
            try:
                tenant = _validated_id(request.get("tenant_id"), "tenant_id")
                job = _validated_id(request.get("job_id"), "job_id")
            except ValueError as exc:
                raise _RpcRejection("ERR_INVALID_FRAME", str(exc)) from exc
            fence = request.get("fencing_token")
            if type(fence) is not int or fence < 1:
                raise _RpcRejection("ERR_INVALID_FRAME", "Positive fencing token required")
            if not isinstance(request.get("inputs"), dict) or not isinstance(request.get("config"), dict):
                raise _RpcRejection("ERR_INVALID_FRAME", "Inputs and config must be objects")
            mode = request.get("fidelity_mode")
            if mode not in ("DOC", "EDITORIAL", "CREATIVE", "IDENTITY_REVIEW"):
                raise _RpcRejection("ERR_INVALID_FRAME", "Unknown fidelity mode")
            seed = request.get("seed")
            if type(seed) is not int or seed < 0:
                raise _RpcRejection("ERR_INVALID_FRAME", "Unsigned integer seed required")
            ctx = ExecutionContext(job, tenant, fence, mode, seed,
                                   self.cas.root_dir, self.cas.tmp_dir)
            result = node.execute(ctx, request["inputs"], request["config"])
            if not isinstance(result, dict):
                raise _RpcRejection("ERR_INVALID_OUTPUT", "Operator output must be an object")
            return self._response({"status": "OK", "abi_version": OMEGA_ABI_VERSION,
                                   "job_id": job, "fencing_token": fence,
                                   "outputs": result})
        except _RpcRejection as exc:
            return self._response({"status": "ERROR", "code": exc.code, "message": str(exc)})
        except Exception as exc:
            # No filesystem paths, traceback or secrets in a client-visible response.
            return self._response({"status": "ERROR", "code": "ERR_OPERATOR_FAILED",
                                   "message": type(exc).__name__})

    @staticmethod
    def _read_exact(connection: socket.socket, count: int) -> bytes | None:
        data = bytearray()
        while len(data) < count:
            piece = connection.recv(count - len(data))
            if not piece:
                return None
            data.extend(piece)
        return bytes(data)

    def handle_connection(self, connection: socket.socket) -> None:
        header = self._read_exact(connection, 4)
        if header is None:
            return
        length = struct.unpack(">I", header)[0]
        if length > MAX_RPC_FRAME_BYTES:
            response = self._response({"status": "ERROR", "code": "ERR_FRAME_TOO_LARGE"})
        else:
            frame = self._read_exact(connection, length)
            response = (self.handle_frame(frame) if frame is not None else
                        self._response({"status": "ERROR", "code": "ERR_TRUNCATED_FRAME"}))
        connection.sendall(struct.pack(">I", len(response)) + response)

    def serve_forever(self) -> None:
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise FileExistsError(self.socket_path)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(self.socket_path))
            os.chmod(self.socket_path, stat.S_IRUSR | stat.S_IWUSR)
            server.listen(16)
            try:
                while True:
                    client, _ = server.accept()
                    with client:
                        client.settimeout(30)
                        try:
                            self.handle_connection(client)
                        except (OSError, TimeoutError):
                            continue
            finally:
                if self.socket_path.is_socket():
                    self.socket_path.unlink()
