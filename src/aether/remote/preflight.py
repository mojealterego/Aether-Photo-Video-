"""Fail-closed ComfyUI compatibility check from live /object_info/{class}.

No local model installation is implied. This runs only when the real network
client is used, before any tile upload/dispatch.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote


def verify_remote_graph(client:Any, graph:dict[str,dict]) -> dict:
    if not isinstance(graph,dict) or not graph:
        raise ValueError('Empty ComfyUI graph')
    metadata={}
    for node in graph.values():
        name=node.get('class_type')
        if name in metadata:
            continue
        if not isinstance(name,str) or not name or '/' in name:
            raise ValueError('Bad ComfyUI node class name')
        body=client.json('/object_info/'+quote(name,safe=''))
        if not isinstance(body,dict) or not isinstance(body.get(name),dict):
            raise RuntimeError(f'Missing ComfyUI node class on server: {name}')
        info=body[name]
        if not isinstance(info.get('input'),dict) or not isinstance(info.get('output'),list):
            raise RuntimeError(f'Invalid remote ComfyUI node metadata for {name}')
        metadata[name]=info
    for uid,node in graph.items():
        name=node['class_type']
        allowed=metadata[name]['input']
        accepted=set(allowed.get('required',{})) | set(allowed.get('optional',{})) | set(allowed.get('hidden',{}))
        inputs=node.get('inputs')
        if not isinstance(inputs,dict) or not set(inputs).issubset(accepted):
            raise RuntimeError(f'Incompatible remote node input schema: {name} ({uid})')
        for k in allowed.get('required',{}):
            if k not in inputs:
                raise RuntimeError(f'Missing required ComfyUI input {name}.{k}')
        for key,field in inputs.items():
            # Class-loader widget selections are supplied by /object_info on server.
            if name in ('CheckpointLoaderSimple','ControlNetLoader') and key in ('ckpt_name','control_net_name'):
                param=allowed.get('required',{}).get(key)
                if isinstance(param,(list,tuple)) and param and isinstance(param[0],(list,tuple)):
                    if field not in param[0]:
                        raise RuntimeError(f'Model checkpoint not installed on remote ComfyUI: {field}')
            if isinstance(field,list):
                if len(field)!=2 or str(field[0]) not in graph or type(field[1]) is not int:
                    raise RuntimeError('Malformed graph link')
                src=graph[str(field[0])]['class_type']
                outputs=metadata[src]['output']
                if field[1] >=len(outputs) or field[1]<0:
                    raise RuntimeError(f'Output port not supported by server: {src}')
    return {'checked_classes':len(metadata), 'node_count':len(graph),
            'server_verified':True}