"""ComfyUI tile journal and transport guard tests with a fake server."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from io import BytesIO
from aether.remote.transaction import execute_tile,ComfyHttp

class Fake:
    def __init__(self):
        self.count=0
        self.output=None
        self.fail=False
    def submit(self,graph):
        self.count+=1
        if self.fail:raise TimeoutError('Ambiguous POST')
        return 'p-1'
    def history(self,pid):
        if self.output is None:return {}
        return {pid:{'outputs':{'3':{'images':[{'filename':'done.png','type':'output'}]}}}}
    def fetch_png(self,record):
        return self.output

def valid_png():
    io=BytesIO()
    Image.new('RGB',(8,8),(44,55,66)).save(io,'PNG')
    return io.getvalue()

class ComfyTransactionTests(unittest.TestCase):
    def test_reject_plain_http_except_loopback(self):
        with self.assertRaises(ValueError):ComfyHttp('http://example.org:8188')
        self.assertIsInstance(ComfyHttp('http://127.0.0.1:8188'),ComfyHttp)

    def test_one_submission_is_reused_after_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);server=Fake();server.output=valid_png()
            args=dict(client=server,graph={'1':{'class_type':'LoadImage','inputs':{}}},
                      identity='t_001',state_dir=d,output=d/'out.png',output_node='3',
                      expected_size=(8,8),input_sha256='a'*64)
            self.assertEqual(execute_tile(**args)['state'],'done')
            self.assertEqual(execute_tile(**args)['state'],'done')
            self.assertEqual(server.count,1)

    def test_uncertain_post_is_never_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);server=Fake();server.fail=True
            args=dict(client=server,graph={'1':{}},identity='t_001',state_dir=d,
                      output=d/'out.png',output_node='3',expected_size=(8,8),
                      input_sha256='a'*64)
            with self.assertRaises(TimeoutError):execute_tile(**args)
            with self.assertRaisesRegex(RuntimeError,'Ambiguous'):execute_tile(**args)
            self.assertEqual(server.count,1)

    def test_known_prompt_can_resume_without_resubmit(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);server=Fake()
            args=dict(client=server,graph={'1':{}},identity='t_001',state_dir=d,
                      output=d/'out.png',output_node='3',expected_size=(8,8),
                      input_sha256='a'*64,max_polls=1)
            with self.assertRaises(TimeoutError):execute_tile(**args)
            server.output=valid_png()
            self.assertEqual(execute_tile(**args)['state'],'done')
            self.assertEqual(server.count,1)

    def test_bad_remote_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);server=Fake();server.output=b'not a png'
            args=dict(client=server,graph={'1':{}},identity='t_001',state_dir=d,
                      output=d/'out.png',output_node='3',expected_size=(8,8),
                      input_sha256='a'*64)
            with self.assertRaises(Exception):execute_tile(**args)
            self.assertFalse((d/'out.png').exists())

    def test_input_sha_changes_block_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);server=Fake();server.output=valid_png()
            args=dict(client=server,graph={'1':{}},identity='t_001',state_dir=d,
                      output=d/'out.png',output_node='3',expected_size=(8,8),
                      input_sha256='a'*64)
            execute_tile(**args)
            args['input_sha256']='b'*64
            with self.assertRaisesRegex(ValueError,'changed'):execute_tile(**args)

if __name__=='__main__':unittest.main()
