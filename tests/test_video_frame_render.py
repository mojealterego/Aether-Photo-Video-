"""Test frame-render checkpoint reuse and source integrity."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np,tifffile
from aether.video.frame_pipeline import _sha
from aether.video.frame_render import process_video_frames

class TestVideoFrameRender(unittest.TestCase):
    def test_processing_recovery_and_integrity(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);source=root/'frames';source.mkdir()
            original=root/'original.mkv';original.write_bytes(b'synthetic metadata only')
            entries=[]
            for i in range(2):
                name=f'frame_{i:06d}.tif'
                image=np.full((12,16,3),100+i,dtype='u2')
                tifffile.imwrite(source/name,image,photometric='rgb')
                entries.append({'name':name,'sha256':_sha(source/name)})
            meta={'version':1,'source_sha256':_sha(original),'max_frames':100,
                  'info':{'width':16,'height':12,'fps':'2','audio':False},'frames':entries}
            (source/'manifest.json').write_text(json.dumps(meta))
            calls=[]
            def fake_render(inp,out,geometry,tile):
                calls.append(inp)
                tifffile.imwrite(out,tifffile.imread(inp),photometric='rgb',
                                  metadata=None,bigtiff=True)
            target=root/'restored'
            x=process_video_frames(source,target,width=16,height=12,tile=32,
                                   render_frame=fake_render)
            self.assertEqual(x['frames'],2)
            self.assertEqual(len(calls),2)
            y=process_video_frames(source,target,width=16,height=12,tile=32,
                                   render_frame=fake_render)
            self.assertEqual(y['reused'],2)
            self.assertEqual(len(calls),2)
            (target/'frame_000000.tif').unlink()
            with self.assertRaises(ValueError):
                process_video_frames(source,target,width=16,height=12,tile=32,
                                     render_frame=fake_render)

    def test_reject_bad_source_manifest(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);source=root/'frames';source.mkdir()
            (source/'manifest.json').write_text('{}')
            with self.assertRaises(ValueError):
                process_video_frames(source,root/'out',width=16,height=12)
if __name__=='__main__':unittest.main()
