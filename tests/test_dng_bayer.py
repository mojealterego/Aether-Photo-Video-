"""Synthetic CFA Bayer 16-bit DNG contract tests."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
from aether.ingest.dng import decode_uncompressed_dng,RAWUnsupportedError,XYZ_TO_SRGB

def make_dng(path,scene,pattern=(0,1,1,2)):
    h,w=scene.shape[:2]
    bayer=np.asarray(pattern).reshape(2,2)
    sensor=np.empty((h,w),'u2')
    for y in range(h):
        for x in range(w):
            sensor[y,x]=round(64+4031*float(scene[y,x,bayer[y%2,x%2]]))
    tags=[(33421,'H',2,(2,2),False),(33422,'B',4,pattern,False),
          (50714,'H',1,64,False),(50717,'H',1,4095,False),
          (50728,'d',3,(1.,1.,1.),False),
          (50721,'d',9,tuple(float(v) for v in XYZ_TO_SRGB.ravel()),False)]
    tifffile.imwrite(path,sensor,photometric='cfa',extratags=tags,metadata=None)

class TestDNG(unittest.TestCase):
    def test_known_flat_scene_and_rgb16(self):
        with tempfile.TemporaryDirectory() as p:
            d=Path(p);src=d/'x.dng';out=d/'y.tif'
            scene=np.broadcast_to(np.array([.3,.5,.7],'f4'),(35,47,3))
            make_dng(src,scene)
            result=decode_uncompressed_dng(src,out,tile_size=16)
            self.assertFalse(result['full_dng_coverage'])
            with tifffile.TiffFile(out) as f:
                self.assertTrue(f.is_bigtiff)
                im=f.asarray().astype('f4')/65535
                exp=np.where(scene<=.0031308,12.92*scene,1.055*scene**(1/2.4)-.055)
                self.assertEqual(im.shape,scene.shape)
                np.testing.assert_allclose(im[4:-4,4:-4],exp[4:-4,4:-4],atol=.015)

    def test_true_sixteen_bit_levels(self):
        with tempfile.TemporaryDirectory() as p:
            d=Path(p);src=d/'x.dng';out=d/'y.tif'
            make_dng(src,np.broadcast_to(np.array([.401,.402,.403],'f4'),(16,18,3)))
            decode_uncompressed_dng(src,out,tile_size=16)
            self.assertTrue(np.any(tifffile.imread(out)%257!=0))

    def test_reject_rgb_disguised_as_dng(self):
        with tempfile.TemporaryDirectory() as p:
            d=Path(p);src=d/'x.dng';out=d/'y.tif'
            tifffile.imwrite(src,np.ones((12,14,3),'u2'),photometric='rgb')
            with self.assertRaises(RAWUnsupportedError):
                decode_uncompressed_dng(src,out)
            self.assertFalse(out.exists())

    def test_reject_non_bayer_and_missing_calibration(self):
        with tempfile.TemporaryDirectory() as p:
            d=Path(p);src=d/'x.dng';out=d/'y.tif'
            make_dng(src,np.ones((12,12,3),'f4')*.5,pattern=(0,1,2,2))
            with self.assertRaises(RAWUnsupportedError):
                decode_uncompressed_dng(src,out)
            src.unlink()
            tifffile.imwrite(src,np.ones((12,12),'u2')*500,photometric='cfa',
                extratags=[(33421,'H',2,(2,2),False),(33422,'B',4,(0,1,1,2),False)])
            with self.assertRaises(RAWUnsupportedError):
                decode_uncompressed_dng(src,out)

if __name__=='__main__':unittest.main()
