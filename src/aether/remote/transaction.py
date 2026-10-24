"""Fail-closed, synchronous ComfyUI single-tile job transaction for S24 Ultra.

An ambiguous POST is persisted as "submitting" and is never retried
automatically. The remote worker is optional; no GPU is used on Android.
"""
from __future__ import annotations
from io import BytesIO
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit,quote
from urllib.request import Request,urlopen
from PIL import Image

def digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def write_state(path:Path,state:dict)->None:
    data=json.dumps(state,sort_keys=True,indent=2,ensure_ascii=False).encode('utf-8')
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as out:
        out.write(data);out.flush();os.fsync(out.fileno())
    os.replace(tmp,path)

class ComfyHttp:
    def __init__(self,base_url:str,*,timeout:float=45.):
        url=urlsplit(base_url)
        if (url.scheme not in ('http','https') or url.username or url.password
            or not url.hostname or url.query or url.fragment
            or (url.scheme=='http' and url.hostname not in ('localhost','127.0.0.1','::1'))):
            raise ValueError('ComfyUI must use HTTPS or loopback HTTP')
        if not 1<=timeout<=600:raise ValueError('Bad HTTP timeout')
        self.base=base_url.rstrip('/')
        self.timeout=timeout

    def json(self,path:str,post:dict|None=None)->dict:
        if not path.startswith('/') or path.startswith('//') or '://' in path:
            raise ValueError('Expected local ComfyUI endpoint path')
        body=json.dumps(post).encode('utf-8') if post is not None else None
        req=Request(self.base+path,data=body,
                    headers={'Content-Type':'application/json'} if body else {},
                    method='POST' if body is not None else 'GET')
        with urlopen(req,timeout=self.timeout) as response:
            data=response.read(4*1024*1024+1)
        if len(data)>4*1024*1024:raise ValueError('ComfyUI JSON is oversized')
        decoded=json.loads(data)
        if not isinstance(decoded,dict):raise ValueError('ComfyUI returned non-object')
        return decoded

    def submit(self,graph:dict)->str:
        response=self.json('/prompt',{'prompt':graph})
        pid=response.get('prompt_id')
        if not isinstance(pid,str) or not pid or len(pid)>128:
            raise RuntimeError('ComfyUI prompt_id missing')
        return pid

    def history(self,prompt_id:str)->dict:
        return self.json('/history/'+quote(prompt_id,safe=''))

    def fetch_png(self,record:dict)->bytes:
        name=record.get('filename')
        folder=record.get('subfolder','')
        kind=record.get('type','output')
        if (not isinstance(name,str) or not name or len(name)>240
            or '/' in name or chr(92) in name or not name.lower().endswith('.png')
            or not isinstance(folder,str) or '/' in folder or chr(92) in folder or folder in ('.','..')
            or kind!='output'):
            raise ValueError('Unexpected remote image descriptor')
        url=(self.base+'/view?filename='+quote(name,safe='')+
             '&subfolder='+quote(folder,safe='')+'&type=output')
        with urlopen(Request(url,method='GET'),timeout=self.timeout) as response:
            blob=response.read(48*1024*1024+1)
        if len(blob)>48*1024*1024:raise ValueError('Remote tile PNG >48 MiB')
        return blob

def _validated_png(data:bytes,size:tuple[int,int])->None:
    if not data or len(data)>48*1024*1024:raise ValueError('Invalid PNG byte size')
    with Image.open(BytesIO(data)) as img:
        if img.format!='PNG' or img.mode!='RGB' or img.size!=size:
            raise ValueError('Expected RGB8 PNG candidate at exact dimensions')
        img.verify()

def execute_tile(*,client:Any,graph:dict,identity:str,state_dir:Path,
                 output:Path,output_node:str,expected_size:tuple[int,int],
                 input_sha256:str,max_polls:int=6,poll_interval:float=1.)->dict:
    """Run or resume one tile with durable SHA binding and uncertain-POST veto."""
    if (not identity or len(identity)>96 or
        not all(x.isalnum() or x in '_-' for x in identity)
        or not isinstance(graph,dict) or not graph
        or not 1<=max_polls<=500 or not 0<=poll_interval<=120
        or not isinstance(input_sha256,str) or len(input_sha256)!=64
        or any(x not in '0123456789abcdef' for x in input_sha256)):
        raise ValueError('Bad tile job or SHA configuration')
    w,h=expected_size
    if not 1<=w<=2048 or not 1<=h<=2048:
        raise ValueError('Neural tile exceeds budget')
    state_dir=Path(state_dir)
    output=Path(output)
    state_dir.mkdir(parents=True,exist_ok=True)
    state_path=state_dir/(identity+'.json')
    signature=digest(json.dumps({'graph':graph,'identity':identity,
        'size':[w,h],'node':output_node,'input_sha256':input_sha256},
        sort_keys=True).encode('utf-8'))
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding='utf-8'))
        if state.get('signature')!=signature:
            raise ValueError('Job settings/input changed during resume')
    else:
        if output.exists():raise FileExistsError('Untracked output')
        state={'signature':signature,'state':'pending',
               'prompt_id':None,'output_sha256':None}
        write_state(state_path,state)
    if state['state']=='done':
        if (not output.is_file() or output.stat().st_size>48*1024*1024
            or digest(output.read_bytes())!=state['output_sha256']):
            raise ValueError('Completed output SHA mismatch')
        _validated_png(output.read_bytes(),(w,h))
        return state
    if state['state']=='submitting':
        raise RuntimeError('Ambiguous remote POST, no automatic resubmit')
    if state['state']=='pending':
        state['state']='submitting'
        write_state(state_path,state)
        # Failure/timeout here preserves the ambiguous state.
        prompt_id=client.submit(graph)
        state['prompt_id']=prompt_id
        state['state']='submitted'
        write_state(state_path,state)
    if state['state']!='submitted':raise ValueError('Unknown journal status')
    prompt_id=state['prompt_id']
    for n in range(max_polls):
        if n and poll_interval:time.sleep(poll_interval)
        history=client.history(prompt_id)
        record=history.get(prompt_id)
        if record is None:continue
        if not isinstance(record,dict):
            raise RuntimeError('Malformed ComfyUI history')
        status=record.get('status',{})
        if isinstance(status,dict) and status.get('status_str')=='error':
            raise RuntimeError('Remote ComfyUI job failed')
        outputs=record.get('outputs',{})
        node=outputs.get(str(output_node),{}) if isinstance(outputs,dict) else {}
        images=node.get('images') if isinstance(node,dict) else None
        if images:
            if len(images)!=1:raise ValueError('Expected exactly one tile')
            blob=client.fetch_png(images[0])
            _validated_png(blob,(w,h))
            if output.exists():raise FileExistsError('Untracked output appeared')
            output.parent.mkdir(parents=True,exist_ok=True)
            temp=output.with_name(output.name+'.partial')
            with temp.open('xb') as f:
                f.write(blob);f.flush();os.fsync(f.fileno())
            os.replace(temp,output)
            state.update(state='done',output_sha256=digest(blob))
            write_state(state_path,state)
            return state
    raise TimeoutError('Remote tile incomplete; resume same prompt_id')
