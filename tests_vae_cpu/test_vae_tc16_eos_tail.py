"""Regression for optional, trainable TC16 incomplete EOS latent."""
import unittest
import torch
from omega_sdk.vae.temporal_tc16 import TemporalVAE16, EncoderState, DecoderState


class TC16TailTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(2)
        self.model = TemporalVAE16(
            base_channels=4, latent_channels=16, model_id='eos-v2',
            encode_eos_tail=True).eval()

    def test_exact_tail_geometry(self):
        for frames, latents in [(1, 1), (2, 2), (3, 2), (4, 2),
                                (5, 2), (6, 3), (8, 3), (9, 3),
                                (10, 4), (11, 4), (12, 4), (13, 4)]:
            with self.subTest(frames=frames):
                x = torch.rand((1, 3, frames, 16, 16))
                z = self.model.encode(x).mode()
                self.assertEqual(z.shape[2], latents)
                y = self.model.decode(z, target_frames=frames, output_hw=(16, 16))
                self.assertEqual(y.shape, x.shape)

    def test_chunk_alignment_with_final_partial(self):
        x = torch.rand((1, 3, 10, 16, 16))
        whole = self.model.encode(x).mode()
        state = EncoderState(self.model.model_id)
        chunks = []
        for idx, (a, b) in enumerate([(0, 2), (2, 4), (4, 9), (9, 10)]):
            post, state = self.model.encode_chunk(
                x[:, :, a:b], state, finalize=(idx == 3))
            chunks.append(post.mode())
        z = torch.cat(chunks, dim=2)
        self.assertTrue(torch.allclose(whole, z, atol=1e-6))
        decoded_whole = self.model.decode(z, target_frames=10, output_hw=(16, 16))
        dstate = DecoderState(self.model.model_id)
        part_a, dstate = self.model.decode_chunk(
            z[:, :, :2], dstate, output_hw=(16, 16))
        part_b, dstate = self.model.decode_chunk(
            z[:, :, 2:], dstate, finalize=True,
            target_frames=10, output_hw=(16, 16))
        self.assertTrue(torch.allclose(
            decoded_whole, torch.cat((part_a, part_b), dim=2), atol=1e-6))

    def test_last_source_frame_changes_final_latent(self):
        source = torch.zeros((1, 3, 10, 16, 16))
        before = self.model.encode(source).mode()
        source[:, :, -1] = 1
        after = self.model.encode(source).mode()
        self.assertFalse(torch.allclose(before[:, :, -1], after[:, :, -1]))

    def test_legacy_checkpoint_and_eos_policy(self):
        old = TemporalVAE16(
            base_channels=4, latent_channels=16, model_id='eos-v2')
        self.model.load_state_dict(old.state_dict(), strict=True)
        self.assertFalse(old.encode_eos_tail)
        sample = torch.rand((1, 3, 10, 16, 16))
        self.assertEqual(old.encode(sample).mode().shape[2], 3)
        self.assertEqual(self.model.encode(sample).mode().shape[2], 4)


if __name__ == '__main__':
    unittest.main()
