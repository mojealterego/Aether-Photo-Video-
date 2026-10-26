"""Omega VAE-017 resource plan and mathematical full-frame tile coverage."""
import unittest
from dataclasses import replace
from omega_sdk.vae.tiling_plan import TilePlanner, TilingError, TilePlan, Tile

class TilingPlanTests(unittest.TestCase):
    def test_400mp_plan_has_bounded_memory_without_full_frame_allocation(self):
        plan=TilePlanner.build(24495,16329,tile_core=512,halo=64,vram_budget_bytes=8<<30)
        self.assertEqual(plan.image_size,(24495,16329))
        self.assertGreater(len(plan.tiles),1000)
        self.assertTrue(TilePlanner.verify_coverage(plan))
        self.assertEqual(plan.latent_shape,(1,16,1,3062,2042))
        self.assertEqual(plan.latent_bytes,1*16*1*3062*2042*2)
        self.assertLess(plan.estimated_peak_vram_bytes,8<<30)

    def test_rejects_insufficient_vram_with_suggestion(self):
        with self.assertRaisesRegex(TilingError,'suggested_core='):
            TilePlanner.build(1000,1000,tile_core=512,halo=64,vram_budget_bytes=100<<20)

    def test_odd_dimensions_and_border_halo(self):
        plan=TilePlanner.build(17,19,tile_core=8,halo=8,vram_budget_bytes=100<<20,base_channels=8)
        self.assertTrue(TilePlanner.verify_coverage(plan))
        self.assertTrue(any(t.core_y1==17 and t.core_x1==19 for t in plan.tiles))
        for t in plan.tiles:
            self.assertGreater(t.padded_y1-t.padded_y0,0)
            self.assertTrue(t.padded_y0<=t.core_y0<t.core_y1<=t.padded_y1)

    def test_missing_tile_or_overlap_core_rejected(self):
        plan=TilePlanner.build(64,64,tile_core=32,halo=8,vram_budget_bytes=1<<30)
        self.assertTrue(TilePlanner.verify_coverage(plan))
        with self.assertRaises(TilingError):TilePlanner.verify_coverage(replace(plan,tiles=plan.tiles[:-1]))
        t=plan.tiles[0]
        broken=replace(t,core_x1=t.core_x1+1)
        with self.assertRaises(TilingError):TilePlanner.verify_coverage(replace(plan,tiles=(broken,)+plan.tiles[1:]))

    def test_mismatched_dimension_limits_rejected(self):
        for args in [(0,100),(100,0),(-1,10),(2**34,2**34)]:
            with self.subTest(args=args),self.assertRaises(TilingError):
                TilePlanner.build(*args,vram_budget_bytes=8<<30)

    def test_rejects_unaligned_halo_or_nonlinear_seed_policy(self):
        with self.assertRaises(TilingError):TilePlanner.build(100,100,tile_core=33,halo=8,vram_budget_bytes=1<<30)
        with self.assertRaises(TilingError):TilePlanner.build(100,100,tile_core=32,halo=5,vram_budget_bytes=1<<30)

    def test_tile_grid_coverage_detected_without_dense_frame_weights(self):
        plan=TilePlanner.build(24495,16329,tile_core=512,halo=64,vram_budget_bytes=8<<30)
        self.assertFalse(hasattr(plan,'weight_map'))
        self.assertTrue(TilePlanner.verify_coverage(plan))

if __name__=='__main__':unittest.main()
