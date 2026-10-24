import tempfile
import unittest
from pathlib import Path
import numpy as np
from aether.core.psf_calibration import measure_point_source_psf,save_psf_calibration,calibrate_radial_ca
from aether.vision.arcface_r100 import ArcFaceR100,align_face_112,verify_arcface_pair

class FakeIO:
    def __init__(self,name,shape=None): self.name=name; self.shape=shape

class FakeSession:
    def get_inputs(self):return [FakeIO("data",[1,3,112,112])]
    def get_outputs(self):return [FakeIO("embedding")]
    def run(self,names,data):return [np.ones((1,512),dtype=np.float32)]

class TestCalibrationArcface(unittest.TestCase):
    def test_psf(self):
        grid=np.ones((9,9),dtype=np.float32)
        grid[4,4]=99
        k=measure_point_source_psf(grid)
        self.assertAlmostEqual(float(k.sum()),1.,places=6)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"calibration.npy"
            save_psf_calibration(path,k)
            np.testing.assert_array_equal(np.load(path,allow_pickle=False),k)

    def test_radial_ca_zero(self):
        p=np.array([[20,20],[60,30],[200,80],[400,200],[700,100],
                    [900,400],[1200,300],[1400,600]],dtype=np.float64)
        self.assertAlmostEqual(calibrate_radial_ca(p,p,(1600,900)),0.,places=7)

    def test_arcface_embedding_requires_real_checkpoint_and_session(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"licensed-r100.onnx"
            path.write_bytes(b"test-only; fake session")
            adapter=ArcFaceR100(path,session=FakeSession())
            aligned=np.zeros((112,112,3),dtype=np.uint8)
            vector=adapter.encode_aligned(aligned)
            self.assertEqual(vector.shape,(512,))
            self.assertAlmostEqual(float(np.linalg.norm(vector)),1,places=5)
            accepted,sim=verify_arcface_pair(vector,vector,threshold=.8)
            self.assertTrue(accepted)
            self.assertAlmostEqual(sim,1.,places=5)

    def test_arcface_invalid_landmarks_fail(self):
        photo=np.zeros((120,120,3),dtype=np.uint8)
        with self.assertRaises(ValueError):
            align_face_112(photo,np.zeros((5,2),dtype=np.float32))

if __name__=="__main__":unittest.main()
