"""ComfyUI stage chaining with an injected fake worker: no real GPU proof."""
import tempfile,unittest
from pathlib import Path
from io import BytesIO
from PIL import Image
from aether.remote.gpu_acceptance import run_tile_stages,validate_remote_url
def png():
    buf=BytesIO();Image.new('RGB',(32,24),(80,90,100)).save(buf,format='PNG')
    return buf.getvalue()
class Fake:
    def __init__(self):self.submissions=0;self.uploads=0
    def upload_png(self,path):self.uploads+=1;return 'uploaded.png'
    def json(self,path,post=None):
        if path=='/system_stats':return {'devices':[{'type':'cuda','name':'FAKE'}]}
        if path.startswith('/object_info/'):
            node=path.split('/')[-1]
            allowed={'LoadImage':['image'],'Passthrough':['image'],'SaveImage':['images']}
            return {node:{'input':{'required':{k:['STRING',{}] for k in allowed[node]}},'output':['IMAGE']}}
        raise ValueError(path)
    def submit(self,graph):self.submissions+=1;return 'p'+str(self.submissions)
    def history(self,pid):return {pid:{'outputs':{'3':{'images':[{'filename':'out.png','type':'output'}]}}}}
    def fetch_png(self,record):return png()
class TestGPUAcceptance(unittest.TestCase):
    def test_https_or_loopback_only(self):
        with self.assertRaises(ValueError):validate_remote_url('http://public.example:8188')
        validate_remote_url('https://owned-worker.example')
    def test_macro_meso_micro_single_tile(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root);input=d/'tile.png';input.write_bytes(png())
            graph={'1':{'class_type':'LoadImage','inputs':{'image':'tile.png'}},
                   '2':{'class_type':'Passthrough','inputs':{'image':['1',0]}},
                   '3':{'class_type':'SaveImage','inputs':{'images':['2',0]}}}
            fake=Fake()
            r=run_tile_stages(fake,input,d/'work',stages={'macro':graph,'meso':graph,'micro':graph},
                              input_node='1',output_node='3')
            self.assertEqual(len(r['completed']),3)
            self.assertEqual(fake.submissions,3)
            self.assertEqual(fake.uploads,3)
    def test_gpu_missing_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root);p=d/'tile.png';p.write_bytes(png())
            class NoGPU(Fake):
                def json(self,path,post=None):
                    return {'devices':[]} if path=='/system_stats' else super().json(path,post)
            with self.assertRaises(RuntimeError):
                run_tile_stages(NoGPU(),p,d/'work',stages={'macro':{}},input_node='1')
if __name__=='__main__':unittest.main()
