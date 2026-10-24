"""ONNX forward-pass validators are contract-tested with fake sessions only."""
import unittest
import numpy as np
from aether.vision.model_acceptance import validate_session,check_evidence

class Descriptor:
    def __init__(self,shape):self.name='rgb';self.shape=shape
class Session:
    def __init__(self,input_shape,output_shape):
        self.shape=input_shape;self.output_shape=output_shape;self.calls=0
    def get_inputs(self):return [Descriptor(self.shape)]
    def get_outputs(self):return [Descriptor(self.output_shape)]
    def run(self,names,inputs):
        self.calls+=1
        return [np.ones(self.output_shape,dtype='f4')]

class TestNeuralForward(unittest.TestCase):
    def test_depth_forward_not_just_metadata(self):
        fake=Session([1,3,64,64],[1,1,64,64])
        self.assertEqual(validate_session(fake,'depth')['status'],'executed')
        self.assertEqual(fake.calls,1)
    def test_arcface_embedding(self):
        fake=Session([1,3,112,112],[1,512])
        self.assertEqual(validate_session(fake,'arcface')['output_shape'],[1,512])
    def test_codeformer_export_contract(self):
        fake=Session([1,3,512,512],[1,3,512,512])
        self.assertEqual(validate_session(fake,'codeformer')['status'],'executed')
    def test_wrong_embedding_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_session(Session([1,3,112,112],[1,64]),'arcface')
    def test_no_false_evidence(self):
        with self.assertRaises(ValueError):check_evidence({'status':'not_run'},required=['depth'])
    def test_recorded_real_execution_evidence(self):
        self.assertTrue(check_evidence({'depth':{'status':'executed',
            'provider':'CPUExecutionProvider','model_sha256':'a'*64}},required=['depth']))

if __name__=='__main__':unittest.main()
