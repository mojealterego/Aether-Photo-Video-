"""Opt-in actual ComfyUI GPU tile E2E, with SHA-256 verified stage chaining.

Uses existing fail-closed transaction and remote /object_info preflight.
Tests only one bounded tile; 400MP assembly remains on Samsung S24.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request,urlopen
from PIL import Image
from .transaction import ComfyHttp,execute_tile
from .preflight import verify_remote_graph

def validate_remote_url(url:str)->None:
    u=urlsplit(url)
    if (u.scheme not in ('http','https') or not u.hostname or u.username or u.password
        or u.query or u.fragment
        or (u.scheme=='http' and u.hostname not in ('127.0.0.1','localhost','::1'))):
        raise ValueError('Require HTTPS or loopback-only HTTP')

class ComfyTileClient:
    def __init__(self,base_url:str,timeout:int=120):
        validate_remote_url(base_url)
        self.http=ComfyHttp(base_url,timeout=timeout)
    def json(self,*a,**kw):return self.http.json(*a,**kw)
    def submit(self,*a,**kw):return self.http.submit(*a,**kw)
    def history(self,*a,**kw):return self.http.history(*a,**kw)
    def fetch_png(self,*a,**kw):return self.http.fetch_png(*a,**kw)
    def upload_png(self,path:Path)->str:
        path=Path(path)
        if path.stat().st_size>48*1024*1024:raise ValueError('Oversized ComfyUI tile')
        data=path.read_bytes()
        with Image.open(BytesIO(data)) as img:
            if img.format!='PNG' or img.mode!='RGB' or max(img.size)>2048:
                raise ValueError('Bounded RGB8 PNG required')
            img.verify()
        name=path.name
        if '/' in name or chr(92) in name:raise ValueError('Unsafe upload filename')
        boundary='AetherTileUpload5fae2d'
        prefix=(f'--{boundary}\r\nContent-Disposition: form-data; '
                f'name="image"; filename="{name}"\r\n'
                'Content-Type: image/png\r\n\r\n').encode()
        body=prefix+data+f'\r\n--{boundary}--\r\n'.encode()
        request=Request(self.http.base+'/upload/image',data=body,
            headers={'Content-Type':f'multipart/form-data; boundary={boundary}'},method='POST')
        with urlopen(request,timeout=self.http.timeout) as result:
            response=json.loads(result.read(65536))
        filename=response.get('name')
        if not isinstance(filename,str) or not filename or '/' in filename or chr(92) in filename or len(filename)>240:
            raise ValueError('Invalid ComfyUI uploaded filename')
        return filename

def run_tile_stages(client,source_tile:Path,workdir:Path,*,stages:dict,
                    input_node:str='10',output_node:str='30',max_polls:int=120,
                    require_gpu:bool=True)->dict:
    path=Path(source_tile)
    with Image.open(path) as img:
        if img.format!='PNG' or img.mode!='RGB' or max(img.size)>2048:
            raise ValueError('Only bounded RGB PNG input')
        wh=img.size
        img.verify()
    if not isinstance(stages,dict) or not stages or len(stages)>4:
        raise ValueError('One to four stage API graphs required')
    stats=client.json('/system_stats')
    if require_gpu and not stats.get('devices'):
        raise RuntimeError('No GPU device advertised by worker')
    work=Path(workdir);work.mkdir(parents=True,exist_ok=True)
    done=[];tile=path
    for stage,template in stages.items():
        if stage not in ('inpaint','macro','meso','micro'):
            raise ValueError('Unknown stage')
        graph=deepcopy(template)
        node=graph.get(str(input_node))
        if not isinstance(node,dict) or node.get('class_type')!='LoadImage':
            raise ValueError('Specified input node must be LoadImage')
        remote_name=client.upload_png(tile)
        node['inputs']['image']=remote_name
        verify_remote_graph(client,graph)
        candidate=work/f'{stage}_candidate.png'
        inp_hash=hashlib.sha256(tile.read_bytes()).hexdigest()
        saved=execute_tile(client=client,graph=graph,identity=f'gpu_{stage}',
            state_dir=work/'ledger',output=candidate,output_node=str(output_node),
            expected_size=wh,input_sha256=inp_hash,max_polls=max_polls,poll_interval=.5)
        if saved.get('state')!='done':raise RuntimeError('Missing remote candidate')
        done.append({'stage':stage,'file':str(candidate),'sha256':saved['output_sha256']})
        tile=candidate
    proof={'status':'actual_remote_result_received','completed':done,
           'gpu_advertised':bool(stats.get('devices')),
           'scope':'one_tile_only_not_400mp','execution_device':'remote_gpu'}
    document=work/'gpu_evidence.json'
    if document.exists():raise FileExistsError('Evidence already exists')
    partial=document.with_name(document.name+'.partial')
    with partial.open('x',encoding='utf8') as out:
        json.dump(proof,out,indent=2,sort_keys=True)
        out.flush();os.fsync(out.fileno())
    os.replace(partial,document)
    return proof

def main(argv=None)->int:
    p=argparse.ArgumentParser(description='Real optional ComfyUI tile GPU acceptance')
    p.add_argument('--server',required=True)
    p.add_argument('--tile',type=Path,required=True)
    p.add_argument('--graphs',type=Path,required=True,help='JSON object keyed inpaint/macro/meso/micro')
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--input-node',default='10')
    p.add_argument('--output-node',default='30')
    a=p.parse_args(argv)
    result=run_tile_stages(ComfyTileClient(a.server),a.tile,a.workdir,
        stages=json.loads(a.graphs.read_text()),input_node=a.input_node,
        output_node=a.output_node)
    print(json.dumps(result,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
