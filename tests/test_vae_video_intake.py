"""Real-file video intake acceptance. No training consent from media upload."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from omega_sdk.vae.video_intake import audit_files, audit_video, tc16_time_contract, MediaIntakeError


class TC16Contract(unittest.TestCase):
    def test_anchor_group_and_tail_invariants(self):
        for frames, latents, tail in [(1,1,0),(2,1,1),(4,1,3),(5,2,0),(8,2,3),
                                       (9,3,0),(168,42,3),(192,48,3),(240,60,3)]:
            with self.subTest(frames=frames):
                contract=tc16_time_contract(frames)
                self.assertEqual(contract['latent_frames'],latents)
                self.assertEqual(contract['tail_frames_not_encoded'],tail)
                self.assertEqual(contract['decoded_from_groups']+tail,frames)
                self.assertFalse(contract['bit_exact_tail_restoration'])

    def test_bad_count_refused(self):
        for n in (0,-1,1.0,True,'192'):
            with self.subTest(n=n),self.assertRaises(MediaIntakeError):tc16_time_contract(n)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg tools required')
class RealContainerIntake(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clip=Path(self.tmp.name)/'clip.mp4'
        subprocess.run(['ffmpeg','-nostdin','-loglevel','error','-f','lavfi',
                        '-i','testsrc2=size=64x64:rate=24', '-frames:v','11',
                        '-c:v','mpeg4','-y',str(self.clip)],check=True,timeout=20)

    def test_real_file_preflight_and_read_only(self):
        before=hashlib.sha256(self.clip.read_bytes()).hexdigest()
        audit=audit_video(self.clip,verify_decode=True)
        after=hashlib.sha256(self.clip.read_bytes()).hexdigest()
        self.assertEqual(before,after)
        self.assertEqual(audit['sha256'],before)
        self.assertEqual(audit['frames'],11)
        self.assertEqual(audit['width'],64)
        self.assertEqual(audit['height'],64)
        self.assertEqual(audit['tc16']['latent_frames'],3)
        self.assertEqual(audit['tc16']['tail_frames_not_encoded'],2)
        self.assertEqual(audit['decode_check'],'PASS')
        self.assertFalse(audit['training_approved'])
        self.assertEqual(audit['permitted_use'],'VALIDATION_ONLY')

    def test_one_clip_and_multiple_paths(self):
        index=audit_files([self.clip])
        self.assertEqual(index['summary']['clips'],1)
        self.assertEqual(index['summary']['frames'],11)
        self.assertEqual(index['summary']['tc16_tail_frames_not_encoded'],2)
        self.assertEqual(index['training_permission'],'NOT_GRANTED')

    def test_cli_writes_manifest_exclusively_and_never_overwrites(self):
        output = Path(self.tmp.name)/'intake.json'
        command = ['python','-m','omega_sdk.vae.video_intake',
                   '--output',str(output),str(self.clip)]
        first = subprocess.run(command,capture_output=True,text=True,timeout=20)
        self.assertEqual(first.returncode,0,first.stderr)
        import json
        data=json.loads(output.read_text())
        self.assertEqual(data['training_permission'],'NOT_GRANTED')
        second = subprocess.run(command,capture_output=True,text=True,timeout=20)
        self.assertNotEqual(second.returncode,0)
        self.assertEqual(json.loads(output.read_text())['summary']['frames'],11)

    def test_duplicate_source_is_rejected(self):
        with self.assertRaises(MediaIntakeError):audit_files([self.clip,self.clip])

    def test_bad_video_container_is_rejected(self):
        fake=Path(self.tmp.name)/'bad.mp4'
        fake.write_text('not a video')
        with self.assertRaises(MediaIntakeError):audit_video(fake)

    def test_unexpected_extension_is_rejected(self):
        fake=Path(self.tmp.name)/'bad.txt'
        fake.write_bytes(self.clip.read_bytes())
        with self.assertRaises(MediaIntakeError):audit_video(fake)


if __name__ == '__main__':unittest.main()
