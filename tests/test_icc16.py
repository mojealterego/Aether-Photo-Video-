"""16-bit ICC regression tests: preserve precision and fail closed."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
from aether.ingest.icc16 import ICCTransform, srgb_profile, rgb16_tiff_to_srgb

class TestICC16(unittest.TestCase):
    def test_identity_roundtrip_preserves_sub_8bit_levels(self):
        image=np.array([[[0,1,256],[32769,40123,65535]],[[4567,34567,12345],[65535,10001,29999]]],dtype='u2')
        with ICCTransform(srgb_profile()) as cms: result=cms.convert(image)
        np.testing.assert_allclose(result.astype('i4'),image.astype('i4'),atol=32)
        self.assertNotEqual(int(result[1,1,1]),int((image[1,1,1]//257)*257))
    def test_real_rgb16_bigtiff(self):
        with tempfile.TemporaryDirectory() as root:
            src=Path(root)/'input.tif';out=Path(root)/'output.tif'
            image=np.random.default_rng(4).integers(0,65536,size=(67,53,3),dtype='u2')
            tifffile.imwrite(src,image,photometric='rgb',metadata=None,iccprofile=srgb_profile())
            details=rgb16_tiff_to_srgb(src,out,rows=7)
            self.assertTrue(details['input_icc_present'])
            with tifffile.TiffFile(out) as t:
                self.assertTrue(t.is_bigtiff)
                np.testing.assert_allclose(t.asarray().astype('i4'),image.astype('i4'),atol=32)
    def test_no_icc_requires_explicit_assumption(self):
        with tempfile.TemporaryDirectory() as root:
            src=Path(root)/'in.tif';out=Path(root)/'out.tif'
            tifffile.imwrite(src,np.full((17,17,3),137,dtype='u1'),photometric='rgb',metadata=None)
            with self.assertRaisesRegex(ValueError,'No source ICC'):
                rgb16_tiff_to_srgb(src,out)
            rgb16_tiff_to_srgb(src,out,assume_srgb=True)
            np.testing.assert_equal(tifffile.imread(out),137*257)
    def test_exif_orientation_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            src=Path(root)/'in.tif';out=Path(root)/'out.tif'
            tifffile.imwrite(src,np.zeros((12,12,3),'u2'),photometric='rgb',
                             extratags=[(274,'H',1,6,False)])
            with self.assertRaisesRegex(ValueError,'orientation'):
                rgb16_tiff_to_srgb(src,out,assume_srgb=True)
            self.assertFalse(out.exists())

if __name__=='__main__': unittest.main()
