"""Offline S24 focus-to-BigTIFF regression contracts; no neural/GPU required."""
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import tifffile

from mobile_s24.auto_focus import FocusConfig, prepare_focus_mask, run_offline_auto


def make_source(path):
    a = np.full((64,80,3), 125, dtype=np.uint8)
    yy,xx = np.indices((30,30))
    a[15:45,22:52] = np.where((xx+yy)%2==0, 20, 240)[:,:,None]
    Image.fromarray(a,'RGB').save(path)


class TestMobileAutoFocus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'input.png'
        make_source(self.source)

    def tearDown(self):
        self.tmp.cleanup()

    def test_focus_proposal_detects_texture(self):
        mask = prepare_focus_mask(FocusConfig(self.source,self.root/'work',preview_limit=128))
        with Image.open(mask) as im:
            self.assertEqual(im.mode,'L')
            arr = np.asarray(im)
            self.assertGreater(float(arr[30,35]),float(arr[5,5])+30)

    def test_sha_resume_and_changed_input_veto(self):
        cfg = FocusConfig(self.source,self.root/'work')
        a = prepare_focus_mask(cfg)
        self.assertEqual(a,prepare_focus_mask(cfg))
        pix = np.asarray(Image.open(self.source)).copy()
        pix[0,0,0] = 10
        Image.fromarray(pix,'RGB').save(self.source)
        with self.assertRaisesRegex(ValueError,'signature'):
            prepare_focus_mask(cfg)

    def test_small_bigtiff_end_to_end(self):
        report = run_offline_auto(FocusConfig(self.source,self.root/'work'),
                                  output=self.root/'result.tif',target_width=97,
                                  target_height=71,tile=32,reserve_gb=0)
        self.assertEqual(report['focus_method'],'global_edge_energy_preview')
        with tifffile.TiffFile(self.root/'result.tif') as tf:
            self.assertTrue(tf.is_bigtiff)
            self.assertEqual(tf.pages[0].shape,(71,97,3))
            self.assertEqual(tf.pages[0].dtype,np.dtype('uint16'))
            im = tf.pages[0].asarray()
            self.assertGreater(float(im.max()),float(im.min()))

    def test_optional_relative_depth_registration(self):
        path = self.root/'relative_depth.png'
        depth = np.zeros((20,24),dtype=np.uint8)
        depth[5:15,4:20] = 127
        Image.fromarray(depth,'L').save(path)
        cfg = FocusConfig(self.source,self.root/'work',depth_map=path)
        self.assertTrue(prepare_focus_mask(cfg).exists())
        data = json.loads((self.root/'work'/'focus_manifest.json').read_text())
        self.assertIsNotNone(data['depth_sha256'])

    def test_depth_requires_grayscale(self):
        with self.assertRaisesRegex(ValueError,'depth'):
            prepare_focus_mask(FocusConfig(self.source,self.root/'work',depth_map=self.source))

    def test_no_opencv_dependency(self):
        with patch.dict(sys.modules, {'cv2':None}):
            mask = prepare_focus_mask(FocusConfig(self.source,self.root/'work'))
            self.assertTrue(mask.exists())


if __name__=='__main__':
    unittest.main()
