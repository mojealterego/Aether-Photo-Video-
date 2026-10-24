"""Real FFmpeg frame and audio round-trip tests (Linux/Termux FFmpeg needed)."""
import shutil,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np,tifffile
from aether.video.frame_pipeline import (
    inspect_video,extract_frames_rgb16,mux_frames_lossless,VideoUnsupported)

@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg not installed')
class VideoTests(unittest.TestCase):
    def test_rgb16_round_trip_with_audio(self):
        with tempfile.TemporaryDirectory() as p:
            d=Path(p);src=d/'source.mkv';out=d/'frames';dst=d/'output.mkv'
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error',
                '-f','lavfi','-i','testsrc2=size=64x48:rate=3:duration=1',
                '-f','lavfi','-i','sine=frequency=440:duration=1',
                '-map','0:v','-map','1:a','-c:v','ffv1',
                '-c:a','pcm_s16le',str(src)],check=True)
            self.assertTrue(inspect_video(src)['audio'])
            self.assertEqual(extract_frames_rgb16(src,out,max_frames=5)['frames'],3)
            for f in sorted(out.glob('frame_*.tif')):
                img=tifffile.imread(f)
                self.assertEqual(img.shape,(48,64,3))
                self.assertEqual(img.dtype,np.dtype('uint16'))
            report=mux_frames_lossless(out,src,dst)
            self.assertEqual(report['video_codec'],'ffv1')
            self.assertTrue(inspect_video(dst)['audio'])

    def test_missing_frame_fails_resume(self):
        with tempfile.TemporaryDirectory() as p:
            d=Path(p);src=d/'video.mkv'
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error',
                '-f','lavfi','-i','color=c=red:s=32x32:r=2:d=1',
                '-c:v','ffv1',str(src)],check=True)
            extract_frames_rgb16(src,d/'frames',max_frames=4)
            sorted((d/'frames').glob('frame_*.tif'))[0].unlink()
            with self.assertRaises(ValueError):
                extract_frames_rgb16(src,d/'frames',max_frames=4)

    def test_pixel_limit_preflight(self):
        with tempfile.TemporaryDirectory() as p:
            d=Path(p);src=d/'video.mkv'
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error',
                '-f','lavfi','-i','color=s=64x48:r=1:d=1',
                '-c:v','ffv1',str(src)],check=True)
            with self.assertRaises(VideoUnsupported):
                extract_frames_rgb16(src,d/'frames',max_pixels=2)

if __name__=='__main__':unittest.main()
