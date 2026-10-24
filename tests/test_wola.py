"""Contract tests for bounded, disk-backed Gaussian WOLA."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from aether.core.wola import WolaMemmap

class TestWolaMemmap(unittest.TestCase):
    def test_constant_image_stays_constant_on_overlap(self):
        with tempfile.TemporaryDirectory() as p:
            with WolaMemmap(Path(p), width=9, height=6) as writer:
                writer.add(np.full((6, 6, 3), .4, dtype=np.float32), x0=0, y0=0)
                writer.add(np.full((6, 6, 3), .4, dtype=np.float32), x0=3, y0=0)
                output = np.concatenate(list(writer.iter_normalized_rows(stripe_rows=2)), axis=0)
                self.assertEqual(output.shape, (6, 9, 3))
                np.testing.assert_allclose(output, .4, atol=1e-6)

    def test_overlap_is_weighted_average_not_repeated_division(self):
        with tempfile.TemporaryDirectory() as p:
            with WolaMemmap(Path(p), width=4, height=3) as writer:
                writer.add(np.full((3, 4, 3), .2, dtype=np.float32), x0=0, y0=0)
                writer.add(np.full((3, 4, 3), .8, dtype=np.float32), x0=0, y0=0)
                output = np.concatenate(list(writer.iter_normalized_rows()), axis=0)
                np.testing.assert_allclose(output, .5, atol=1e-6)

    def test_uncovered_pixel_fails_closed(self):
        with tempfile.TemporaryDirectory() as p:
            with WolaMemmap(Path(p), width=3, height=3) as writer:
                writer.add(np.ones((2, 2, 3), np.float32), x0=0, y0=0)
                with self.assertRaises(ValueError):
                    list(writer.iter_normalized_rows())

    def test_reject_invalid_tile_and_coordinates(self):
        with tempfile.TemporaryDirectory() as p:
            with WolaMemmap(Path(p), width=4, height=4) as writer:
                with self.assertRaises(ValueError):
                    writer.add(np.ones((4, 4, 3), np.float32), x0=1, y0=1)
                with self.assertRaises(ValueError):
                    writer.add(np.ones((2, 2, 1), np.float32), x0=0, y0=0)
                with self.assertRaises(ValueError):
                    writer.add(np.ones((2, 2, 3), np.float32) * np.nan, x0=0, y0=0)

    def test_output_generator_is_striped_rgb16(self):
        with tempfile.TemporaryDirectory() as p:
            with WolaMemmap(Path(p), width=7, height=5) as writer:
                writer.add(np.ones((5, 7, 3), np.float32), x0=0, y0=0)
                blocks = list(writer.iter_uint16_rows(stripe_rows=2))
                self.assertEqual([b.shape[0] for b in blocks], [2, 2, 1])
                self.assertTrue(all(b.dtype == np.uint16 for b in blocks))
                np.testing.assert_equal(np.concatenate(blocks, axis=0), 65535)

if __name__ == '__main__':
    unittest.main()
