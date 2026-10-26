"""OVLT v1 bounded authenticated image and video latent transport contracts."""
import unittest
import numpy as np
from pathlib import Path
import tempfile
from omega_sdk.vae.contracts import LatentSchema
from omega_sdk.vae.ovlt import OVLTv1,OVLTError

class TransportTests(unittest.TestCase):
    def setUp(self):
        self.schema=LatentSchema('P16','enc','dec',16)
        self.a=np.random.default_rng(3).random((1,16,4,5),dtype=np.float32)
        self.key=b'k'*32
        self.model='a'*64
        self.source='b'*64
    def encode(self,data=None):return OVLTv1.serialize(
        self.a if data is None else data,self.schema,self.model,source_sha256=self.source,auth_key=self.key)
    def test_image_round_trip(self):
        payload=self.encode()
        result,meta=OVLTv1.deserialize(payload,expected_schema=self.schema,expected_model_sha256=self.model,auth_key=self.key)
        np.testing.assert_array_equal(result,self.a)
        self.assertEqual(meta['schema_sha256'],self.schema.digest())
    def test_video_round_trip(self):
        v=np.stack((self.a,self.a+3,self.a+6),axis=2)
        payload=self.encode(v)
        result,_=OVLTv1.deserialize(payload,expected_schema=self.schema,expected_model_sha256=self.model,auth_key=self.key)
        np.testing.assert_array_equal(result,v)
    def test_cross_model_and_schema_refused(self):
        payload=self.encode()
        with self.assertRaises(OVLTError):OVLTv1.deserialize(payload,expected_schema=self.schema,expected_model_sha256='0'*64,auth_key=self.key)
        bad=LatentSchema('P16','foreign','dec',16)
        with self.assertRaises(OVLTError):OVLTv1.deserialize(payload,expected_schema=bad,expected_model_sha256=self.model,auth_key=self.key)
    def test_tamper_and_wrong_key_fail_closed(self):
        payload=bytearray(self.encode());payload[-1]^=0x3
        with self.assertRaises(OVLTError):OVLTv1.deserialize(bytes(payload),expected_schema=self.schema,expected_model_sha256=self.model,auth_key=self.key)
        with self.assertRaises(OVLTError):OVLTv1.deserialize(self.encode(),expected_schema=self.schema,expected_model_sha256=self.model,auth_key=b'x'*32)
    def test_missing_key_refused(self):
        with self.assertRaises(OVLTError):OVLTv1.serialize(self.a,self.schema,self.model,source_sha256=self.source,auth_key=b'default')
    def test_file_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'latent.ovlt';OVLTv1.write_exclusive(p,self.encode())
            with self.assertRaises(FileExistsError):OVLTv1.write_exclusive(p,b'override')
if __name__=='__main__':unittest.main()
