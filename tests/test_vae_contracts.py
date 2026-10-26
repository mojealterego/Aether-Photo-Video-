"""AETHERIS Ω-VAE unit acceptance for contracts, Ω-R and TR-0."""
import json, struct, hashlib, unittest
from dataclasses import replace
import numpy as np
from omega_sdk.vae.contracts import LatentSchema, ColorContract, ModelProfile, SafeTensorHeaderValidator, ContractError, normalize_latent, denormalize_latent
from omega_sdk.vae.reversible import encode_residual, decode_residual, ReversibleError
from omega_sdk.vae.governance import SampleRecord, verify_dataset, GovernanceError

class Contracts(unittest.TestCase):
    def test_five_profiles(self):
        self.assertEqual(set(x.value for x in ModelProfile), {'P16','G16','TC16','E8','R'})
    def test_latent_sha_and_normalization_roundtrip(self):
        s=LatentSchema('P16','enc1','dec1',16,means=(.5,)*16,scales=(2.,)*16)
        z=np.arange(16*2*3,dtype=np.float32).reshape(1,16,2,3)
        np.testing.assert_array_equal(denormalize_latent(normalize_latent(z,s),s),z)
        self.assertEqual(s.digest(),s.digest())
        self.assertEqual(len(s.digest()),71)
        self.assertNotEqual(replace(s,encoder_id='enc2').digest(),s.digest())
    def test_reject_cross_pair_and_nan_scales(self):
        with self.assertRaises(ContractError):LatentSchema('P16','a','b',16,scales=(float('nan'),)*16)
        with self.assertRaises(ContractError):LatentSchema('P16','a','b',16,scales=(0.,)*16)
        with self.assertRaises(ContractError):LatentSchema('P16','a','b',16,means=(0.,)*14)
    def test_reject_wrong_dtype_or_shape(self):
        s=LatentSchema('P16','a','b',16)
        with self.assertRaises(ContractError):normalize_latent(np.zeros((1,15,2,2),dtype='f4'),s)
        with self.assertRaises(ContractError):normalize_latent(np.zeros((1,16,2,2),dtype='f8'),s)
    def test_color_digest_validates_parameters(self):
        c=ColorContract()
        self.assertEqual(len(c.digest()),71)
        with self.assertRaises(ContractError):replace(c,exposure_scale=-1).digest()
        with self.assertRaises(ContractError):replace(c,working_primaries='').digest()
    def test_safetensors_accepts_complete_header(self):
        hdr={'my.weight':{'dtype':'F32','shape':[2],'data_offsets':[0,8]},'__metadata__':{'format':'pt'}}
        raw=json.dumps(hdr,separators=(',',':')).encode()
        payload=struct.pack('<Q',len(raw))+raw+bytes(8)
        self.assertEqual(SafeTensorHeaderValidator.inspect(payload)['tensor_count'],1)
    def test_safetensors_rejects_gap_overlap_bad_size_and_nan(self):
        records=[
            {'x':{'dtype':'U8','shape':[1],'data_offsets':[1,2]}},
            {'x':{'dtype':'U8','shape':[2],'data_offsets':[0,1]}},
            {'x':{'dtype':'F32','shape':[1],'data_offsets':[0,8]}},
            {'x':{'dtype':'U8','shape':[1],'data_offsets':[0,1]},'y':{'dtype':'U8','shape':[1],'data_offsets':[0,1]}},
            {'x':{'dtype':'U8','shape':[1],'data_offsets':[0,1]},'__metadata__':{'bad':float('nan')}}
        ]
        for header in records:
            raw=json.dumps(header,allow_nan=True).encode()
            with self.subTest(header=header),self.assertRaises(ContractError):
                SafeTensorHeaderValidator.inspect(struct.pack('<Q',len(raw))+raw+bytes(2))
        with self.assertRaises(ContractError):SafeTensorHeaderValidator.inspect(struct.pack('<Q',100*1024*1024+1))
    def test_safetensors_rejects_unhashable_dtype_with_contract_error(self):
        raw=json.dumps({'x':{'dtype':[],'shape':[1],'data_offsets':[0,1]}}).encode()
        with self.assertRaises(ContractError):
            SafeTensorHeaderValidator.inspect(struct.pack('<Q',len(raw))+raw+b'x')

    def test_safetensors_rejects_duplicate_json_keys(self):
        raw=b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        with self.assertRaises(ContractError):SafeTensorHeaderValidator.inspect(struct.pack('<Q',len(raw))+raw+b'1')
    def test_residual_bit_exact_boundary_values(self):
        source=np.array([0,65535,32768,50],dtype=np.uint16).reshape(2,2)
        base=np.array([65535,0,32768,0],dtype=np.uint16).reshape(2,2)
        packet=encode_residual(source,base)
        output,passed=decode_residual(packet)
        np.testing.assert_array_equal(output,source)
        self.assertTrue(passed)
        self.assertEqual(output.dtype,np.uint16)
    def test_residual_tamper_and_decompression_bomb_are_blocked(self):
        source=np.arange(256,dtype=np.uint16).reshape(16,16)
        packet=encode_residual(source,np.zeros_like(source))
        with self.assertRaises(ReversibleError):decode_residual(replace(packet,residual_sha256='0'*64))
        with self.assertRaises(ReversibleError):decode_residual(replace(packet,shape=(9999,9999)))
    def test_residual_missing_sidecar_cannot_claim_lossless(self):
        output,certified=decode_residual(None,lossy_fallback=np.zeros((2,2),dtype=np.uint16))
        self.assertFalse(certified)
        with self.assertRaises(ReversibleError):decode_residual(None)
    def test_dataset_governance_rejects_session_or_subject_leakage(self):
        a=SampleRecord('a','sha256:'+'a'*64,'train','lic',('vae_training',),'NEF',14,'ACEScg','session1','subject1','verified_granted')
        b=replace(a,sample_id='b',immutable_content_sha256='sha256:'+'b'*64,split_id='test')
        with self.assertRaises(GovernanceError):verify_dataset([a,b])
        b=replace(b,scene_or_session_id='session2')
        with self.assertRaises(GovernanceError):verify_dataset([a,b])
        b=replace(b,subject_group_id='subject2')
        self.assertTrue(verify_dataset([a,b]))
    def test_dataset_governance_license_and_consent(self):
        a=SampleRecord('a','sha256:'+'a'*64,'train','lic',('vae_training',),'NEF',14,'ACEScg','s','p','verified_granted')
        for change in ({'consent_status':'revoked'}, {'acquisition_license_id':''}, {'permitted_training_scopes':('other',)}, {'color_profile':''}):
            with self.subTest(change=change),self.assertRaises(GovernanceError):verify_dataset([replace(a,**change)])
if __name__=='__main__':unittest.main()
