"""Numerical tests of the source-preserving multilevel Haar/DWT backend."""
import unittest
import numpy as np
from aether.core.wavelet import haar_decompose,haar_reconstruct,wavelet_faithful_fusion

class TestWavelet(unittest.TestCase):
    def test_invertible_odd_shapes(self):
        for h,w in [(3,3),(8,8),(19,27),(1,13),(102,65)]:
            for levels in [1,2,3]:
                with self.subTest(shape=(h,w),levels=levels):
                    a=np.random.default_rng(h*100+w).random((h,w,3),dtype=np.float32)
                    low,bands=haar_decompose(a,levels=levels)
                    out=haar_reconstruct(low,bands)
                    self.assertEqual(out.shape,a.shape)
                    np.testing.assert_allclose(out,a,atol=6e-7)

    def test_zero_focus_preserves_original(self):
        a=np.random.default_rng(3).random((31,29,3),dtype=np.float32)
        b=np.random.default_rng(4).random((31,29,3),dtype=np.float32)
        out=wavelet_faithful_fusion(a,b,np.zeros((31,29),np.float32),levels=3)
        np.testing.assert_allclose(out,a,atol=6e-7)

    def test_same_base_and_candidate_is_identity(self):
        a=np.random.default_rng(5).random((31,29,3),dtype=np.float32)
        out=wavelet_faithful_fusion(a,a,np.ones((31,29),np.float32),levels=4)
        np.testing.assert_allclose(out,a,atol=6e-7)

    def test_reject_bad_mask(self):
        a=np.zeros((6,7,3),np.float32)
        with self.assertRaises(ValueError):
            wavelet_faithful_fusion(a,a,np.ones((6,6),np.float32))

    def test_fused_detail_changes_focused_area(self):
        a=np.full((64,64,3),.5,np.float32)
        checker=(np.indices((64,64)).sum(axis=0)%2).astype(np.float32)
        b=np.clip(a+(checker*2-1)[:,:,None]*.12,0,1)
        result=wavelet_faithful_fusion(a,b,np.ones((64,64),np.float32),levels=3)
        self.assertGreater(result.std(),.08)
        np.testing.assert_allclose(result.mean(axis=(0,1)),a.mean(axis=(0,1)),atol=1e-6)

    def test_protected_region_stays_source(self):
        a=np.full((16,18,3),.5,np.float32)
        b=np.zeros_like(a)
        b[::2,::2]=1
        result=wavelet_faithful_fusion(a,b,np.ones((16,18),np.float32),
                                       protected_mask=np.ones((16,18),np.float32))
        np.testing.assert_allclose(result,a,atol=6e-7)

if __name__=='__main__':
    unittest.main()
