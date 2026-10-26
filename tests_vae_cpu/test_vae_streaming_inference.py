"""Disk-backed 400MP-capable tile execution: local CPU contract tests."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from omega_sdk.vae.streaming_inference import run_tiled_neural,InferenceError
from omega_sdk.vae.tiling_plan import TilePlanner

class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.src=Path(self.tmp.name)/'src.npy';self.dst=Path(self.tmp.name)/'dst.npy'
        a=np.random.default_rng(2026).normal(size=(33,47,3)).astype('f4')
        np.save(self.src,a)
        self.a=a
        self.plan=TilePlanner.build(33,47,tile_core=16,halo=8,base_channels=8,vram_budget_bytes=1<<30)

    def test_conv3x3_matches_monolithic_exactly(self):
        torch.manual_seed(12)
        model=torch.nn.Conv2d(3,3,3,padding=1,bias=False).eval()
        result=run_tiled_neural(self.src,self.dst,model,self.plan,test_only=True)
        with torch.no_grad():
            ref=model(torch.from_numpy(self.a.transpose(2,0,1).copy())[None]).numpy()[0].transpose(1,2,0)
        np.testing.assert_allclose(np.load(self.dst),ref,rtol=1e-6,atol=1e-6)
        self.assertEqual(result['tiles_completed'],len(self.plan.tiles))
        self.assertEqual(result['device'],'cpu')
        self.assertFalse(result['production_approved'])

    def test_sparse_zero_tiles_match_dense_and_count_skips(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'zeros.npy';dense=Path(d)/'dense.npy';sparse=Path(d)/'sparse.npy'
            x=np.zeros((32,32,3),dtype='f4');x[0,0]=[.2,.3,.4]
            np.save(src,x)
            plan=TilePlanner.build(32,32,tile_core=16,halo=8,base_channels=8,vram_budget_bytes=1<<30)
            model=torch.nn.Conv2d(3,3,1,bias=False).eval()
            with torch.no_grad():
                model.weight.zero_()
                for i in range(3):model.weight[i,i,0,0]=1
            run_tiled_neural(src,dense,model,plan,test_only=True)
            result=run_tiled_neural(src,sparse,model,plan,test_only=True,sparse_zero_tiles=True)
            np.testing.assert_array_equal(np.load(dense),np.load(sparse))
            self.assertGreater(result['zero_tiles_elided'],0)

    def test_fail_closed_without_production_authorization(self):
        with self.assertRaises(InferenceError):
            run_tiled_neural(self.src,self.dst,torch.nn.Identity().eval(),self.plan)
        self.assertFalse(self.dst.exists())

    def test_nonfinite_neural_output_rolls_back(self):
        class Nan(torch.nn.Module):
            def forward(self,x):return x*float('nan')
        with self.assertRaises(InferenceError):
            run_tiled_neural(self.src,self.dst,Nan().eval(),self.plan,test_only=True)
        self.assertFalse(self.dst.exists())

    def test_existing_output_not_overwritten(self):
        self.dst.write_bytes(b'protected')
        with self.assertRaises(FileExistsError):
            run_tiled_neural(self.src,self.dst,torch.nn.Identity().eval(),self.plan,test_only=True)
        self.assertEqual(self.dst.read_bytes(),b'protected')

    def test_mismatched_plan_refused(self):
        wrong=TilePlanner.build(16,16,tile_core=16,halo=8,base_channels=8,vram_budget_bytes=1<<30)
        with self.assertRaises(InferenceError):
            run_tiled_neural(self.src,self.dst,torch.nn.Identity().eval(),wrong,test_only=True)

if __name__=='__main__':unittest.main()
