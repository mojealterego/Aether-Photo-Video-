"""SYS-AETHERIS-VISION-VAE-017: versioned radiometric and latent contracts.

No executable deserialization, no implicit foreign VAE schema conversion.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import json
import math
import re
import struct
from typing import Any
import numpy as np

class ContractError(ValueError):
    pass

class ModelProfile(str,Enum):
    P16='P16'
    G16='G16'
    TC16='TC16'
    E8='E8'
    R='R'

_HASH=re.compile(r'sha256:[a-f0-9]{64}\Z')
_MAX_ELEMENTS=16*1024*1024*1024

def checked_elements(shape, max_elements=_MAX_ELEMENTS):
    if not shape or type(max_elements)!=int or max_elements<=0:
        raise ContractError('invalid tensor shape or limit')
    total=1
    for x in shape:
        if type(x)!=int or x<=0 or total>max_elements//x:
            raise ContractError('invalid tensor dimension or capacity exceeded')
        total*=x
    return total

def canonical_digest(value:Any)->str:
    def reject_invalid(obj):
        if isinstance(obj,float) and not math.isfinite(obj):
            raise ContractError('NaN/Inf in canonical manifest')
        if isinstance(obj,dict):
            for k,v in obj.items():
                if not isinstance(k,str):raise ContractError('non-text manifest key')
                reject_invalid(v)
        if isinstance(obj,(list,tuple)):
            for x in obj:reject_invalid(x)
    reject_invalid(value)
    try:
        wire=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
    except (ValueError,TypeError) as exc:
        raise ContractError('manifest not canonically serializable') from exc
    return 'sha256:'+hashlib.sha256(wire).hexdigest()

@dataclass(frozen=True)
class ColorContract:
    schema_version:str='aethervae.color/1.0'
    color_encoding:str='scene-linear'
    working_primaries:str='ACEScg'
    white_point:str='D60'
    transfer_function:str='signed_log1p'
    exposure_scale:float=1.0
    luminance_y0:float=.18
    bit_depth:int=16
    alpha_mode:str='straight'
    ocio_config_digest:str=''

    def digest(self)->str:
        if (not all(isinstance(getattr(self,key),str) and getattr(self,key)
                    for key in ('schema_version','color_encoding','working_primaries',
                                'white_point','transfer_function','alpha_mode'))
            or not math.isfinite(self.exposure_scale) or self.exposure_scale<=0
            or not math.isfinite(self.luminance_y0) or self.luminance_y0<=0
            or self.bit_depth not in (16,32) or self.alpha_mode not in ('straight','premultiplied')):
            raise ContractError('invalid color profile parameters')
        return canonical_digest(asdict(self))

@dataclass(frozen=True)
class LatentSchema:
    family:str
    encoder_id:str
    decoder_id:str
    channels:int
    means:tuple[float,...]|None=None
    scales:tuple[float,...]|None=None
    schema_version:str='aethervae.latent/1.0'
    spatial_stride:tuple[int,int]=(8,8)
    temporal_stride:int=1
    axes:tuple[str,...]=('batch','channel','height','width')
    precision:str='float32'
    color_input_schema:str='ACEScg_scene_linear'
    model_checkpoint_sha256:str=''

    def __post_init__(self):
        if self.means is None and type(self.channels)==int and 0<self.channels<=128:
            object.__setattr__(self,'means',(0.,)*self.channels)
        if self.scales is None and type(self.channels)==int and 0<self.channels<=128:
            object.__setattr__(self,'scales',(1.,)*self.channels)
        self.validate()

    def validate(self)->None:
        if (self.family not in {x.value for x in ModelProfile}
            or type(self.channels)!=int or not 1<=self.channels<=128
            or not self.encoder_id or not self.decoder_id
            or not isinstance(self.means,tuple) or not isinstance(self.scales,tuple)
            or len(self.means)!=self.channels or len(self.scales)!=self.channels
            or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in self.means)
            or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in self.scales)
            or self.spatial_stride!=(8,8) or self.temporal_stride not in (1,4)
            or self.precision!='float32' or not self.color_input_schema):
            raise ContractError('invalid or incompatible latent schema')
        if self.model_checkpoint_sha256 and not _HASH.fullmatch(self.model_checkpoint_sha256):
            raise ContractError('invalid checkpoint SHA-256')

    def digest(self)->str:
        self.validate()
        return canonical_digest(asdict(self))

def _latent_array(x:np.ndarray,schema:LatentSchema)->np.ndarray:
    schema.validate()
    if not isinstance(x,np.ndarray) or x.dtype!=np.float32 or x.ndim not in (4,5) or x.shape[1]!=schema.channels or not np.isfinite(x).all():
        raise ContractError('finite float32 NCHW/NCTHW tensor with exact latent channels required')
    checked_elements(x.shape,500_000_000)
    return x

def normalize_latent(x:np.ndarray,schema:LatentSchema)->np.ndarray:
    data=_latent_array(x,schema)
    shape=(1,schema.channels)+(1,)*(data.ndim-2)
    return (data-np.asarray(schema.means,dtype='f4').reshape(shape))/np.asarray(schema.scales,dtype='f4').reshape(shape)

def denormalize_latent(x:np.ndarray,schema:LatentSchema)->np.ndarray:
    data=_latent_array(x,schema)
    shape=(1,schema.channels)+(1,)*(data.ndim-2)
    return data*np.asarray(schema.scales,dtype='f4').reshape(shape)+np.asarray(schema.means,dtype='f4').reshape(shape)

class SafeTensorHeaderValidator:
    """Bounded, strict safetensors header inspection without invoking pickle."""
    MAX_HEADER=100*1024*1024
    MAX_DATA=2*1024*1024*1024
    ITEM_SIZES={'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'U64':8,
                'I32':4,'U32':4,'I16':2,'U16':2,'I8':1,'U8':1,'BOOL':1}

    @classmethod
    def inspect(cls,raw:bytes)->dict[str,Any]:
        if not isinstance(raw,bytes) or len(raw)<8:
            raise ContractError('truncated safetensors header')
        h=struct.unpack_from('<Q',raw,0)[0]
        if h<2 or h>cls.MAX_HEADER or h+8>len(raw):
            raise ContractError('unbounded or truncated safetensors header')
        payload_length=len(raw)-8-h
        if payload_length>cls.MAX_DATA:
            raise ContractError('safetensors payload over limit')
        def pairs(items):
            result={}
            for key,value in items:
                if key in result:raise ContractError('duplicate JSON key')
                result[key]=value
            return result
        def reject_constant(text):raise ContractError('nonfinite JSON constant: '+text)
        try:
            header=json.loads(raw[8:8+h].decode('utf8'),object_pairs_hook=pairs,parse_constant=reject_constant)
        except (ValueError,UnicodeDecodeError,TypeError) as exc:
            raise ContractError('malformed safetensors header JSON') from exc
        if not isinstance(header,dict):raise ContractError('tensor dictionary required')
        meta=header.get('__metadata__',{})
        if not isinstance(meta,dict) or any(type(k)!=str or type(v)!=str for k,v in meta.items()):
            raise ContractError('invalid safetensors metadata')
        intervals=[]
        for name,info in header.items():
            if name=='__metadata__':continue
            if not isinstance(name,str) or not name or not isinstance(info,dict) or set(info)!={'dtype','shape','data_offsets'}:
                raise ContractError('invalid tensor metadata')
            dtype=info['dtype'];shape=info['shape'];limits=info['data_offsets']
            if dtype not in cls.ITEM_SIZES or not isinstance(shape,list) or not isinstance(limits,list) or len(limits)!=2:
                raise ContractError('invalid tensor type, shape or offsets')
            if not shape:count=1  # scalar tensors
            else:count=checked_elements(shape,max_elements=cls.MAX_DATA)
            start,end=limits
            if type(start)!=int or type(end)!=int or start<0 or end<start or end>payload_length:
                raise ContractError('invalid tensor range')
            if (end-start)!=count*cls.ITEM_SIZES[dtype]:
                raise ContractError('tensor shape vs byte range mismatch')
            intervals.append((start,end))
        intervals.sort()
        cursor=0
        for start,end in intervals:
            if start!=cursor:raise ContractError('gap or overlap in tensor buffer')
            cursor=end
        if cursor!=payload_length:raise ContractError('trailing undecoded tensor bytes')
        return {'tensor_count':len(intervals),'payload_bytes':payload_length,'metadata':meta}
