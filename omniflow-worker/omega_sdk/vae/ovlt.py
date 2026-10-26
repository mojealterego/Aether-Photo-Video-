"""OVLT v1 transport: bounded, authenticated chunked float32 image/video latents.

Normative SYS-AETHERIS-VISION-VAE-017 semantics: explicit latent schema SHA,
model package SHA, per-chunk CRC32/SHA256 and caller-provided HMAC key.
This is an in-memory bounded v1 stage, not streaming 400MP acceptance.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
from pathlib import Path
import struct
import tempfile
import zlib
import numpy as np
from .contracts import LatentSchema, ContractError, checked_elements, canonical_digest

MAGIC=b'OVLT\x01\x00\x00\x00'
MAX_METADATA=1<<20
MAX_PAYLOAD=256<<20
MAX_CHUNKS=65536
SHA64='0123456789abcdef'

class OVLTError(ValueError):pass

def _json(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def _sha(x):return hashlib.sha256(x).hexdigest()
def _key(value):
    if not isinstance(value,bytes) or len(value)<32:raise OVLTError('32+ byte secret key required')
def _checksum(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in SHA64 for c in value):
        raise OVLTError('exact SHA256 hex digest required')
def _uncompress(z,expected):
    d=zlib.decompressobj()
    try:
        out=d.decompress(z,expected+1)
        if len(out)!=expected or not d.eof or d.unused_data or d.unconsumed_tail:
            raise OVLTError('bounded decompresion size mismatch')
        out+=d.flush()
    except zlib.error as e:raise OVLTError('corrupted chunk compression') from e
    if len(out)!=expected:raise OVLTError('bad decompressed length')
    return out

class OVLTv1:
    @staticmethod
    def serialize(latent:np.ndarray,schema:LatentSchema,model_package_sha256:str,
                  *,source_sha256:str,auth_key:bytes,timebase:tuple[int,int]=(1,24))->bytes:
        _key(auth_key);schema.validate()
        _checksum(model_package_sha256);_checksum(source_sha256)
        if not isinstance(latent,np.ndarray) or latent.dtype!=np.float32 or latent.ndim not in (4,5):
            raise OVLTError('NCHW or NCTHW float32 tensor required')
        if not np.isfinite(latent).all() or latent.shape[1]!=schema.channels:
            raise OVLTError('invalid shape, numerical values or latent channels')
        if latent.ndim==4:
            padded=latent[:,:,None]
        else:padded=latent
        try:
            total=checked_elements(padded.shape,MAX_PAYLOAD//4)
        except ContractError as e:raise OVLTError('tensor too large') from e
        b,c,t,h,w=map(int,padded.shape)
        if t>MAX_CHUNKS:raise OVLTError('too many video chunks')
        n,d=timebase
        if any(type(x)!=int or x<=0 for x in (n,d)) or d>2**31 or n>2**31:
            raise OVLTError('invalid rational timebase')
        chunks=[];table=[];offset=0
        for index in range(t):
            slice_bytes=np.ascontiguousarray(padded[:,:,index],dtype='<f4').tobytes()
            packed=zlib.compress(slice_bytes,6)
            chunks.append(packed)
            table.append({'index':index,'offset':offset,'byte_length':len(packed),
                          'uncompressed_length':len(slice_bytes),
                          'crc32':zlib.crc32(packed)&0xffffffff,
                          'sha256':_sha(packed)})
            offset+=len(packed)
        payload=b''.join(chunks)
        if len(payload)>MAX_PAYLOAD:raise OVLTError('compressed payload over limit')
        base={'ovlt_version':'1.0','schema_sha256':schema.digest(),
              'model_package_sha256':model_package_sha256,'source_sha256':source_sha256,
              'original_ndim':latent.ndim,'dtype':'F32','shape_bcthw':[b,c,t,h,w],
              'timebase_rational':[n,d],'payload_sha256':_sha(payload)}
        table_bytes=_json(table)
        mac=hmac.new(auth_key,_json(base)+table_bytes+payload,hashlib.sha256).hexdigest()
        manifest=_json({**base,'integrity_hmac_sha256':mac})
        if max(len(table_bytes),len(manifest))>MAX_METADATA:raise OVLTError('OVLT header over limit')
        return struct.pack('<8sQQQ',MAGIC,len(manifest),len(table_bytes),t)+manifest+table_bytes+payload

    @staticmethod
    def deserialize(data:bytes,*,expected_schema:LatentSchema,
                    expected_model_sha256:str,auth_key:bytes)->tuple[np.ndarray,dict]:
        _key(auth_key);_checksum(expected_model_sha256)
        if not isinstance(data,bytes) or len(data)<32 or len(data)>MAX_PAYLOAD+2*MAX_METADATA+32:
            raise OVLTError('unbounded or truncated OVLT')
        magic,ml,tl,n=struct.unpack_from('<8sQQQ',data)
        if magic!=MAGIC or not 0<ml<=MAX_METADATA or not 0<tl<=MAX_METADATA or not 0<n<=MAX_CHUNKS:
            raise OVLTError('bad OVLT binary header')
        end=32+ml+tl
        if end>len(data):raise OVLTError('truncated manifest or chunk table')
        def distinct(items):
            result={}
            for key,value in items:
                if key in result:raise OVLTError('duplicate manifest field')
                result[key]=value
            return result
        def deny_const(_):raise OVLTError('NaN/Inf metadata')
        try:
            manifest=json.loads(data[32:32+ml].decode('utf8'),object_pairs_hook=distinct,parse_constant=deny_const)
            table=json.loads(data[32+ml:end].decode('utf8'),object_pairs_hook=distinct,parse_constant=deny_const)
        except (ValueError,UnicodeDecodeError,TypeError) as e:raise OVLTError('invalid OVLT metadata') from e
        if not isinstance(manifest,dict) or not isinstance(table,list) or len(table)!=n:
            raise OVLTError('invalid OVLT metadata structure')
        if set(manifest)!={'ovlt_version','schema_sha256','model_package_sha256','source_sha256',
                           'original_ndim','dtype','shape_bcthw','timebase_rational','payload_sha256',
                           'integrity_hmac_sha256'}:
            raise OVLTError('invalid manifest keys')
        mac=manifest['integrity_hmac_sha256']
        if not isinstance(mac,str) or len(mac)!=64:raise OVLTError('missing authentication tag')
        body=data[end:]
        unsigned=dict(manifest);unsigned.pop('integrity_hmac_sha256')
        computed=hmac.new(auth_key,_json(unsigned)+data[32+ml:end]+body,hashlib.sha256).hexdigest()
        if not hmac.compare_digest(computed,mac):raise OVLTError('invalid OVLT authentication')
        if manifest['ovlt_version']!='1.0' or manifest['schema_sha256']!=expected_schema.digest() or manifest['model_package_sha256']!=expected_model_sha256:
            raise OVLTError('foreign model or latent schema denied')
        if manifest['dtype']!='F32' or manifest['original_ndim'] not in (4,5) or _sha(body)!=manifest['payload_sha256']:
            raise OVLTError('OVLT dtype or payload integrity mismatch')
        shape=manifest['shape_bcthw']
        if not isinstance(shape,list) or len(shape)!=5 or shape[2]!=n or shape[1]!=expected_schema.channels:
            raise OVLTError('invalid latent dimensions')
        try:checked_elements(shape,MAX_PAYLOAD//4)
        except ContractError as e:raise OVLTError('tensor limit exceeded') from e
        b,c,t,h,w=shape
        offset=0;frames=[]
        expected_bytes=b*c*h*w*4
        for i,rec in enumerate(table):
            if not isinstance(rec,dict) or set(rec)!={'index','offset','byte_length','uncompressed_length','crc32','sha256'}:
                raise OVLTError('bad chunk descriptor')
            if type(rec['index'])!=int or rec['index']!=i or type(rec['offset'])!=int or rec['offset']!=offset or type(rec['byte_length'])!=int or rec['byte_length']<1:
                raise OVLTError('disordered/overlapping payload chunks')
            length=rec['byte_length'];offset+=length
            if offset>len(body) or rec['uncompressed_length']!=expected_bytes:
                raise OVLTError('out of bounds chunk')
            chunk=body[rec['offset']:offset]
            if (zlib.crc32(chunk)&0xffffffff)!=rec['crc32'] or _sha(chunk)!=rec['sha256']:
                raise OVLTError('chunk CRC32/SHA256 mismatch')
            decompressed=_uncompress(chunk,expected_bytes)
            frames.append(np.frombuffer(decompressed,dtype='<f4').reshape(b,c,h,w).copy())
        if offset!=len(body):raise OVLTError('trailing bytes after chunk table')
        arr=np.stack(frames,axis=2)
        if not np.isfinite(arr).all():raise OVLTError('nonfinite decoded model latents')
        if manifest['original_ndim']==4:
            if t!=1:raise OVLTError('invalid video temporal dimension for an image')
            arr=arr[:,:,0]
        return arr,manifest

    @staticmethod
    def write_exclusive(path:Path,payload:bytes)->None:
        target=Path(path)
        if target.exists() or target.is_symlink():raise FileExistsError(target)
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,temp=tempfile.mkstemp(dir=target.parent,prefix='.ovlt-',suffix='.part')
        try:
            with os.fdopen(fd,'wb') as f:
                f.write(payload);f.flush();os.fsync(f.fileno())
            os.link(temp,target)  # exclusive publication: never replace private originals
            directory_fd=os.open(target.parent,os.O_RDONLY)
            try:os.fsync(directory_fd)
            finally:os.close(directory_fd)
        finally:
            Path(temp).unlink(missing_ok=True)
