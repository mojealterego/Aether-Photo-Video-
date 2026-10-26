"""TC16 training smoke: rights, SafeTensors checkpoint and non-promotion."""
import tempfile,unittest
from pathlib import Path
import torch
from omega_sdk.vae.temporal_tc16 import TemporalVAE16
from omega_sdk.vae.training_tc16 import train_smoke,TrainingGateError
from omega_sdk.vae.governance import SampleRecord

class TrainingGateTests(unittest.TestCase):
    def test_synthetic_smoke_checkpoint_is_not_approved(self):
        torch.manual_seed(4)
        model=TemporalVAE16(base_channels=4)
        sample=SampleRecord('test','sha256:'+'a'*64,'train','synthetic',('vae_training',),
                            'SYNTHETIC',16,'ACEScg','synthetic_session','synthetic_subject','synthetic_exempt')
        with tempfile.TemporaryDirectory() as d:
            result=train_smoke(model,[(sample,torch.rand(1,3,5,16,16))],Path(d)/'checkpoint.safetensors',steps=2)
            self.assertEqual(result['steps'],2)
            self.assertFalse(result['production_approved'])
            self.assertEqual(result['status'],'SMOKE_UNAPPROVED')
            self.assertTrue((Path(d)/'checkpoint.safetensors').is_file())
            self.assertTrue(result['weights_sha256'].startswith('sha256:'))

    def test_rejects_no_rights(self):
        model=TemporalVAE16(base_channels=4)
        rec=SampleRecord('a','sha256:'+'b'*64,'train','',('vae_training',),
                         'SYNTHETIC',16,'ACEScg','session','person','revoked')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TrainingGateError):
                train_smoke(model,[(rec,torch.rand(1,3,5,16,16))],Path(d)/'bad.safetensors')

    def test_rejects_real_dataset_for_smoke_only(self):
        model=TemporalVAE16(base_channels=4)
        rec=SampleRecord('a','sha256:'+'b'*64,'train','grant',('vae_training',),
                         'NEF',14,'ACEScg','session','person','verified_granted')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TrainingGateError):
                train_smoke(model,[(rec,torch.rand(1,3,5,16,16))],Path(d)/'bad.safetensors')

if __name__=='__main__':unittest.main()
