"""CIEDE2000 relative color evidence requires explicit ICC and JPEG assumptions."""
import tempfile,unittest
from pathlib import Path
import numpy as np,tifffile
from PIL import Image,ImageCms
from omega_sdk.vae.color_measure import measure_preview_deltae,ColorEvidenceError

class ColorEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.tif=self.root/'ref.tif';self.jpg=self.root/'preview.jpg'
        a=np.full((64,64,3),32768,dtype=np.uint16)
        profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        tifffile.imwrite(self.tif,a,photometric='rgb',iccprofile=profile)
        Image.fromarray(np.full((64,64,3),128,dtype=np.uint8),'RGB').save(self.jpg,quality=100)

    def test_real_color_difference_has_provenance(self):
        record=measure_preview_deltae(self.tif,self.jpg,assume_candidate_srgb=True)
        self.assertLess(record['mean_deltaE00'],1.0)
        self.assertEqual(record['pixels_compared'],4096)
        self.assertEqual(record['reference_icc_name'].strip(),'sRGB built-in')
        self.assertFalse(record['absolute_color_accuracy_certified'])

    def test_rejects_missing_icc_reference(self):
        tifffile.imwrite(self.tif,np.ones((16,16,3),dtype=np.uint16),photometric='rgb')
        with self.assertRaises(ColorEvidenceError):
            measure_preview_deltae(self.tif,self.jpg,assume_candidate_srgb=True)

    def test_requires_explicit_jpeg_srgb_assumption(self):
        with self.assertRaises(ColorEvidenceError):
            measure_preview_deltae(self.tif,self.jpg)

if __name__=='__main__':unittest.main()
