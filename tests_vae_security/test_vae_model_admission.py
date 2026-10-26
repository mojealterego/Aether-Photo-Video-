"""Independent Ed25519 production admission for a verified AETHERIS VAE bundle."""
import hashlib, json, struct, tempfile, unittest
from dataclasses import replace
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
from omega_sdk.vae.contracts import LatentSchema, ColorContract
from omega_sdk.vae.model_admission import (BundleManifest,WeightFile, BundleAdmission,
                                             AdmissionError)

class ModelAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        header=json.dumps({'model.weight':{'dtype':'F32','shape':[2], 'data_offsets':[0,8]}}).encode()
        self.weight=self.root/'weights.safetensors'
        self.weight.write_bytes(struct.pack('<Q',len(header))+header+bytes(8))
        self.secret=Ed25519PrivateKey.generate()
        self.public=self.secret.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)
        self.latent=LatentSchema('P16','enc','dec',16)
        self.color=ColorContract()
        self.manifest=BundleManifest(schema_version='aethervae.model_bundle/1.0',
            package_id='AetherVAE-P16',package_semver='1.0.0',release_status='TRAINED_SIGNED_VALIDATED',
            runtime_abi='omega.node.abi.v1.1',latent_schema_sha256=self.latent.digest(),
            color_contract_sha256=self.color.digest(),weights=(WeightFile('encoder','weights.safetensors',
                    'sha256:'+hashlib.sha256(self.weight.read_bytes()).hexdigest()),),
            license_id='CommercialGrant-01',commercial_use_allowed=True,signer_key_id='root-key',signature_hex='')
        self.signed=replace(self.manifest,signature_hex=self.secret.sign(self.manifest.canonical_payload()).hex())
    def gate(self,m=None, trusted=None):
        return BundleAdmission(trusted_signers={'root-key':self.public} if trusted is None else trusted).verify(
            m or self.signed,root=self.root,latent_schema=self.latent,color_contract=self.color,commercial_use=True)
    def test_signed_valid_weight_and_color_schema_admitted(self):
        record=self.gate()
        self.assertEqual(record['package_id'],'AetherVAE-P16')
        self.assertEqual(record['weight_count'],1)
    def test_unsigned_and_modified_manifest_rejected(self):
        for m in (self.manifest,replace(self.signed,license_id='ChangedLicense')):
            with self.subTest(m=m.signature_hex),self.assertRaises(AdmissionError):self.gate(m)
    def test_untrusted_signer_rejected(self):
        with self.assertRaises(AdmissionError):self.gate(trusted={})
    def test_untrained_release_or_commercial_denied(self):
        for m in (replace(self.manifest,release_status='NOT_TRAINED'),
                  replace(self.manifest,commercial_use_allowed=False)):
            signed=replace(m,signature_hex=self.secret.sign(m.canonical_payload()).hex())
            with self.subTest(status=m.release_status),self.assertRaises(AdmissionError):self.gate(signed)
    def test_weight_mutation_does_not_pass_hash(self):
        self.weight.write_bytes(self.weight.read_bytes()[:-1]+b'x')
        with self.assertRaises(AdmissionError):self.gate()
    def test_traversal_and_symlinks_denied_even_with_valid_signature(self):
        evil=replace(self.manifest,weights=(WeightFile('encoder','../weights.safetensors',self.manifest.weights[0].sha256),))
        evil=replace(evil,signature_hex=self.secret.sign(evil.canonical_payload()).hex())
        with self.assertRaises(AdmissionError):self.gate(evil)
        link=self.root/'link.safetensors';link.symlink_to(self.weight)
        m=replace(self.manifest,weights=(WeightFile('encoder','link.safetensors',self.manifest.weights[0].sha256),))
        m=replace(m,signature_hex=self.secret.sign(m.canonical_payload()).hex())
        with self.assertRaises(AdmissionError):self.gate(m)
    def test_wrong_latent_contract_rejected(self):
        foreign=LatentSchema('P16','other','dec',16)
        with self.assertRaises(AdmissionError):BundleAdmission(trusted_signers={'root-key':self.public}).verify(
            self.signed,root=self.root,latent_schema=foreign,color_contract=self.color,commercial_use=True)
    def test_corrupt_safetensors_header_rejected_even_with_matching_hash(self):
        self.weight.write_bytes(b'corrupt-tensors')
        manifest=replace(self.manifest,weights=(WeightFile('encoder','weights.safetensors',
                             'sha256:'+hashlib.sha256(self.weight.read_bytes()).hexdigest()),))
        signed=replace(manifest,signature_hex=self.secret.sign(manifest.canonical_payload()).hex())
        with self.assertRaises(AdmissionError):self.gate(signed)
if __name__=='__main__':unittest.main()
