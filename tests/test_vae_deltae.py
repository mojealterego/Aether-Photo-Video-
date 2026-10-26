"""CIEDE2000 Sharma et al. reference pairs and hue edge cases."""
import unittest
import numpy as np
from omega_sdk.vae.deltae import ciede2000, ColorMetricError

class DeltaE2000Tests(unittest.TestCase):
    def test_sharma_reference_pairs(self):
        # G. Sharma, W. Wu, E. Dalal, Color Research & Application 30 (2005).
        cases=[
          ((50,2.6772,-79.7751),(50,0,-82.7485),2.0425),
          ((50,3.1571,-77.2803),(50,0,-82.7485),2.8615),
          ((50,2.8361,-74.0200),(50,0,-82.7485),3.4412),
          ((50,-1.3802,-84.2814),(50,0,-82.7485),1.0000),
          ((50,-1.1848,-84.8006),(50,0,-82.7485),1.0000),
          ((50,-0.9009,-85.5211),(50,0,-82.7485),1.0000),
          ((50,0,0),(50,-1,2),2.3669),
          ((50,-1,2),(50,0,0),2.3669),
          ((60.2574,-34.0099,36.2677),(60.4626,-34.1751,39.4387),1.2644),
          ((63.0109,-31.0961,-5.8663),(62.8187,-29.7946,-4.0864),1.2630),
        ]
        for left,right,expected in cases:
            with self.subTest(left=left,right=right):
                got=ciede2000(np.asarray(left),np.asarray(right))
                self.assertAlmostEqual(float(got),expected,places=3)

    def test_identity_zero_and_symmetry(self):
        x=np.array([[30.,50.,-30.],[0.,0.,0.]])
        y=np.array([[40.,10.,-10.],[99.,0.,0.]])
        self.assertTrue(np.array_equal(ciede2000(x,x),np.zeros(2)))
        np.testing.assert_allclose(ciede2000(x,y),ciede2000(y,x),atol=1e-10)

    def test_invalid_shape_and_nonfinite_values_refused(self):
        for a in (np.ones((2,4)),np.array([20.,float('nan'),10.]),np.array([20.,5.])):
            with self.subTest(shape=a.shape),self.assertRaises(ColorMetricError):
                ciede2000(a,np.ones(3))

    def test_results_nonnegative_for_random_lab(self):
        r=np.random.default_rng(3)
        x=r.random((12,20,3))*50-10
        y=r.random((12,20,3))*80-20
        result=ciede2000(x,y)
        self.assertEqual(result.shape,(12,20))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result>=0))

if __name__=='__main__':unittest.main()
