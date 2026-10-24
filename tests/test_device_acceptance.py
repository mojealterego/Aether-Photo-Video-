"""Measured S24/Android acceptance runner contracts; these are Linux-only tests."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from aether.mobile.device_acceptance import AcceptanceConfig,read_device_stats,measured_render

class TestAcceptance(unittest.TestCase):
    def test_stats_have_unavailable_fields_explicitly(self):
        state=read_device_stats()
        self.assertIn('thermal_max_c',state)
        self.assertIn('rss_kib',state)

    def test_small_rgb16_render_produces_report(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);src=d/'source.png';out=d/'out.tif';report=d/'report.json'
            Image.fromarray(np.full((20,30,3),150,dtype='u1')).save(src)
            evidence=measured_render(AcceptanceConfig(src,out,d/'work',report,
                                    width=72,height=48,tile=32,reserve_free_gb=0))
            self.assertTrue(out.exists())
            self.assertTrue(report.exists())
            self.assertEqual(evidence['output_shape_hwc'],[48,72,3])
            self.assertTrue(evidence['no_oom_guarantee'])
            self.assertGreaterEqual(evidence['samples'],1)

    def test_paths_must_not_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder)
            with self.assertRaises(ValueError):
                AcceptanceConfig(d/'source.png',d/'same.tif',d/'work',d/'same.tif')

if __name__=='__main__':unittest.main()
