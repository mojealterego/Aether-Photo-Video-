"""Ω-TC16 CPU architecture and streaming acceptance; untrained weights."""
import unittest
import torch
from omega_sdk.vae.temporal_tc16 import TemporalVAE16, TemporalContractError

class TemporalVAEAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(2026)
        cls.model=TemporalVAE16(base_channels=8,latent_channels=16,model_id='tc16-v1').eval()
        cls.video=torch.randn(1,3,10,32,40)

    def test_exact_anchor_stride_and_output_shapes(self):
        with torch.no_grad():
            post=self.model.encode(self.video)
            self.assertEqual(tuple(post.mode().shape),(1,16,3,4,5))
            recon=self.model.decode(post.mode(),target_frames=10,output_hw=(32,40))
        self.assertEqual(tuple(recon.shape),tuple(self.video.shape))
        self.assertTrue(torch.isfinite(recon).all())

    def test_streaming_chunk_boundaries_equivalent_to_one_shot(self):
        with torch.no_grad():
            whole=self.model.encode(self.video).mode()
            state=None;latents=[]
            for a,b in [(0,1),(1,3),(3,4),(4,8),(8,10)]:
                posterior,state=self.model.encode_chunk(self.video[:,:,a:b],state,finalize=b==10)
                if posterior.mode().shape[2]:latents.append(posterior.mode())
            parts=torch.cat(latents,dim=2)
        torch.testing.assert_close(whole,parts,rtol=0,atol=1e-6)
        with torch.no_grad():
            one=self.model.decode(whole,target_frames=10,output_hw=(32,40))
            state=None;pieces=[]
            for a,b in [(0,1),(1,2),(2,3)]:
                result,state=self.model.decode_chunk(whole[:,:,a:b],state,
                    target_frames=10 if b==3 else None,output_hw=(32,40),finalize=b==3)
                pieces.append(result)
            stitched=torch.cat(pieces,dim=2)
        torch.testing.assert_close(one,stitched,rtol=0,atol=1e-6)

    def test_future_group_does_not_change_past_latent_or_decoded_frames(self):
        a=self.video[:,:,:9].clone();b=a.clone();b[:,:,5:]*=500
        with torch.no_grad():
            za=self.model.encode(a).mode();zb=self.model.encode(b).mode()
            torch.testing.assert_close(za[:,:,:2],zb[:,:,:2],rtol=0,atol=0)
            ra=self.model.decode(za,target_frames=9,output_hw=(32,40))
            rb=self.model.decode(zb,target_frames=9,output_hw=(32,40))
            torch.testing.assert_close(ra[:,:,:5],rb[:,:,:5],rtol=0,atol=0)

    def test_partial_tail_has_no_fake_extra_latent(self):
        with torch.no_grad():
            for t in [1,2,3,4,5,6,8,9,10,13]:
                clip=self.video[:,:,:min(t,10)] if t<=10 else torch.randn(1,3,t,32,40)
                post=self.model.encode(clip)
                self.assertEqual(post.mode().shape[2],1+(t-1)//4)
                rec=self.model.decode(post.mode(),target_frames=t,output_hw=(32,40))
                self.assertEqual(rec.shape[2],t)

    def test_eos_tail_when_final_chunk_contains_no_new_latent(self):
        with torch.no_grad():
            full=self.model.encode(self.video).mode()
            first,state=self.model.decode_chunk(full,target_frames=None,output_hw=(32,40),finalize=False)
            empty=full[:,:,:0]
            tail,state=self.model.decode_chunk(empty,state,target_frames=10,output_hw=(32,40),finalize=True)
            self.assertEqual(tuple(tail.shape),(1,3,1,32,40))
            torch.testing.assert_close(tail,first[:,:,-1:],rtol=0,atol=0)

    def test_reject_cross_model_state_and_state_after_final(self):
        with torch.no_grad():
            _,state=self.model.encode_chunk(self.video[:,:,:2],finalize=True)
            with self.assertRaises(TemporalContractError):
                self.model.encode_chunk(self.video[:,:,:1],state)
            other=TemporalVAE16(base_channels=8,model_id='other')
            with self.assertRaises(TemporalContractError):
                other.encode_chunk(self.video[:,:,:1],state)

    def test_scene_cut_reset_and_odd_spatial_dimensions(self):
        x=torch.randn(1,3,9,33,39)
        with torch.no_grad():
            post=self.model.encode(x)
            self.assertEqual(tuple(post.mode().shape),(1,16,3,5,5))
            out=self.model.decode(post.mode(),target_frames=9,output_hw=(33,39))
            self.assertEqual(tuple(out.shape),tuple(x.shape))
            _,state=self.model.encode_chunk(x[:,:,:3])
            post2,state=self.model.encode_chunk(x[:,:,3:9],state,scene_cut=True,finalize=True)
            self.assertEqual(post2.mode().shape[2],2)

    def test_forward_training_gradient_reaches_encoder_and_decoder(self):
        model=TemporalVAE16(base_channels=8,latent_channels=16).train()
        x=torch.randn(1,3,5,16,16)
        out,posterior=model(x)
        loss=(out-x).square().mean()+posterior.kl()*1e-6
        loss.backward()
        self.assertTrue(torch.isfinite(model.stem[0].weight.grad).all())
        self.assertTrue(torch.isfinite(model.out_rgb.weight.grad).all())

if __name__=='__main__':unittest.main()
