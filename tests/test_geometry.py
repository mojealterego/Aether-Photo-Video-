import json
import subprocess
import sys
import unittest

from aether.core.geometry import target_dimensions, iter_tiles, memory_estimate


class TestGeometry(unittest.TestCase):
    def test_400mp_three_by_two_dimensions(self):
        self.assertEqual(target_dimensions(6000, 4000), (24495, 16330))

    def test_invalid_dimensions(self):
        for shape in [(0, 4000), (100, -1), (1, 1)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    target_dimensions(*shape)

    def test_tile_grid_covers_target_exactly(self):
        w, h = 573, 387
        tiles = list(iter_tiles(w, h, tile_size=256, halo=32))
        covered = [[0] * w for _ in range(h)]
        for tile in tiles:
            self.assertLessEqual(tile.read_x0, tile.x0)
            self.assertGreaterEqual(tile.read_x1, tile.x1)
            self.assertLessEqual(tile.read_y0, tile.y0)
            self.assertGreaterEqual(tile.read_y1, tile.y1)
            self.assertLessEqual(tile.x1 - tile.x0, 256)
            for y in range(tile.y0, tile.y1):
                for x in range(tile.x0, tile.x1):
                    covered[y][x] += 1
        self.assertTrue(all(value == 1 for row in covered for value in row))
        self.assertEqual(len(tiles), 6)

    def test_halo_clips_at_image_boundary(self):
        tiles = list(iter_tiles(100, 50, tile_size=256, halo=32))
        self.assertEqual(len(tiles), 1)
        t = tiles[0]
        self.assertEqual((t.read_x0, t.read_y0, t.read_x1, t.read_y1), (0, 0, 100, 50))

    def test_reject_excessive_allocations(self):
        for tile, halo in [(0, 10), (256, -1), (4096, 32), (64, 300)]:
            with self.subTest(tile=tile, halo=halo):
                with self.assertRaises(ValueError):
                    list(iter_tiles(100, 100, tile_size=tile, halo=halo))

    def test_memmap_estimate(self):
        s = memory_estimate(24495, 16330)
        self.assertEqual(s['pixels'], 400003350)
        self.assertEqual(s['rgb16_bytes'], 400003350 * 6)
        self.assertEqual(s['rgb_float32_bytes'], 400003350 * 12)
        self.assertEqual(s['wola_disk_bytes'], 400003350 * 16)

    def test_cli_is_pure_planner(self):
        result = subprocess.run(
            [sys.executable, '-m', 'aether.mobile.cli', 'plan', '--width', '6000', '--height', '4000'],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        self.assertEqual(data['output'], {'width': 24495, 'height': 16330})
        self.assertEqual(data['processing'], 'plan_only')


if __name__ == '__main__':
    unittest.main()
