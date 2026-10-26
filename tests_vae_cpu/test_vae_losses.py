"""Differentiable AETHERIS Ω-VAE training loss test gates (CPU)."""
import unittest
import torch
from omega_sdk.vae.losses import CompositeLossOmega,LossWeights
from omega_sdk.vae.neural import DiagonalGaussian
from omega_sdk.vae.contracts import ContractError

class VaeLossTests(unittest.TestCase):
    def test_photo_training_refuses_adversarial_weight(self):
        with self.assertRaises(ContractError):CompositeLossOmega(LossWeights(adversarial=.1))
    def test_identical_images_no_hdr_edge_or_freq_penalty(self):
        image=torch.randn(1,3,8,8)
        loss=CompositeLossOmega()
        self.assertLess(loss.charbonnier(image,image).abs().item(),1e-7)
        self.assertLess(loss.log_luma(image,image).abs().item(),1e-7)
        self.assertLess(loss.frequency(image,image).abs().item(),1e-7)
    def test_gradients_pass_through_composite_loss(self):
        torch.manual_seed(4)
        x=torch.randn(1,3,8,8,requires_grad=True)
        y=torch.randn(1,3,8,8)
        post=DiagonalGaussian(torch.zeros(1,32,1,1))
        loss,metrics=CompositeLossOmega()(x,y,post)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIn('focal_frequency',metrics)
    def test_invalid_variance_refused(self):
        x=torch.zeros(1,3,8,8)
        with self.assertRaises(ContractError):CompositeLossOmega()(x,x,DiagonalGaussian(torch.zeros(1,32,1,1)),variance=torch.zeros_like(x))
if __name__=='__main__':unittest.main()
