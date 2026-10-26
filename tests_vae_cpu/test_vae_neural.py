"""CPU architectural tests of untrained Ω-VAE blocks — not model-quality acceptance."""
import unittest
import torch
from omega_sdk.vae.neural import VAE2D,DiagonalGaussian,CausalConv3D,CausalStreamState
from omega_sdk.vae.contracts import ContractError

class NeuralTests(unittest.TestCase):
    def test_untrained_vae_p16_shapes_and_uncertainty(self):
        torch.manual_seed(7)
        m=VAE2D(channels=16,base=8,context_dim=8).eval()
        with torch.no_grad():
            posterior,ctx=m.encode(torch.randn(1,3,32,40))
            self.assertEqual(tuple(posterior.mode().shape),(1,16,4,5))
            out,variance=m.decode(posterior.mode(),ctx,True)
        self.assertEqual(tuple(out.shape),(1,3,32,40))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((variance>0).all())

    def test_seeded_posterior_and_kl(self):
        p=DiagonalGaussian(torch.zeros(1,32,2,2))
        self.assertTrue(torch.equal(p.sample(seed=33),p.sample(seed=33)))
        self.assertGreaterEqual(p.kl(free_bits=.01).item(),.16-.00001)

    def test_vae_rejects_wrong_channel(self):
        with self.assertRaises(ContractError):VAE2D(base=8).encode(torch.zeros(1,1,16,16))

    def test_causal_conv_never_sees_future_frame(self):
        torch.manual_seed(1)
        m=CausalConv3D(3,4).eval()
        x=torch.randn(1,3,6,8,8);alt=x.clone();alt[:,:,4:]+=200
        with torch.no_grad():
            a=m(x);b=m(alt)
        torch.testing.assert_close(a[:,:,:4],b[:,:,:4],atol=0,rtol=0)

    def test_causal_conv_chunk_matches_whole_and_resets(self):
        torch.manual_seed(1)
        m=CausalConv3D(3,4).eval()
        x=torch.randn(1,3,6,8,8)
        with torch.no_grad():
            whole=m(x);state=CausalStreamState(m.model_id)
            parts=torch.cat([m(x[:,:,:2],state),m(x[:,:,2:4],state),m(x[:,:,4:],state)],dim=2)
            torch.testing.assert_close(whole,parts,atol=1e-5,rtol=1e-5)
            state.reset()
            torch.testing.assert_close(m(x[:,:,:2],state),m(x[:,:,:2]),atol=0,rtol=0)

    def test_foreign_causal_state_rejected(self):
        with self.assertRaises(ContractError):
            CausalConv3D(3,4)(torch.ones(1,3,2,4,4),CausalStreamState('foreign'))
if __name__=='__main__':unittest.main()
