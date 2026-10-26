"""Ω-R lossless sidecar: verified uint16 pixels with bounded compressed int32 delta.

A deterministic, retained base raster is encoded with the residual; the
neural predictor is never trusted for a bit-exact archival claim.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import zlib
import numpy as np
from .contracts import checked_elements, ContractError

class ReversibleError(ValueError):pass

@dataclass(frozen=True)
class ResidualPacket:
    shape:tuple[int,...]
    base_zlib:bytes
    delta_zlib:bytes
    base_sha256:str
    residual_sha256:str
    source_uint16_sha256:str

MAX_PIXELS=16_777_216  # ROI only until streaming profile lands

def _arr(data):
    if not isinstance(data,np.ndarray) or data.dtype!=np.uint16 or data.ndim not in (2,3):
        raise ReversibleError('uint16 raster expected')
    try: checked_elements(data.shape,MAX_PIXELS)
    except ContractError as e:raise ReversibleError('raster ROI limit exceeded') from e
    return np.ascontiguousarray(data)

def encode_residual(source:np.ndarray,base:np.ndarray)->ResidualPacket:
    a=_arr(source);b=_arr(base)
    if a.shape!=b.shape:raise ReversibleError('base/source shape mismatch')
    raw_a=a.tobytes();raw_b=b.tobytes()
    residual=np.subtract(a.astype('<i4'),b.astype('<i4'),dtype='<i4').tobytes()
    return ResidualPacket(tuple(a.shape),zlib.compress(raw_b,6),zlib.compress(residual,6),
        hashlib.sha256(raw_b).hexdigest(),hashlib.sha256(residual).hexdigest(),
        hashlib.sha256(raw_a).hexdigest())

def _decompress_exact(compressed:bytes,length:int)->bytes:
    if not isinstance(compressed,bytes):raise ReversibleError('invalid compressed payload')
    d=zlib.decompressobj()
    try:
        raw=d.decompress(compressed,length+1)
        if len(raw)!=length or not d.eof or d.unused_data or d.unconsumed_tail:
            raise ReversibleError('oversized, truncated or trailing compressed payload')
        raw+=d.flush()
    except zlib.error as e:
        raise ReversibleError('corrupt compressed data') from e
    if len(raw)!=length:raise ReversibleError('decompressed buffer size mismatch')
    return raw

def decode_residual(packet:ResidualPacket|None,*,lossy_fallback:np.ndarray|None=None)->tuple[np.ndarray,bool]:
    if packet is None:
        if lossy_fallback is None:raise ReversibleError('missing Ω-R sidecar, no fallback')
        return _arr(lossy_fallback).copy(),False
    try:elements=checked_elements(packet.shape,MAX_PIXELS)
    except ContractError as e:raise ReversibleError('invalid sidecar dimensions') from e
    base_raw=_decompress_exact(packet.base_zlib,elements*2)
    delta_raw=_decompress_exact(packet.delta_zlib,elements*4)
    if hashlib.sha256(base_raw).hexdigest()!=packet.base_sha256 or hashlib.sha256(delta_raw).hexdigest()!=packet.residual_sha256:
        raise ReversibleError('base or residual digest mismatch')
    base=np.frombuffer(base_raw,dtype='<u2').astype(np.int32)
    delta=np.frombuffer(delta_raw,dtype='<i4')
    restored=base.astype(np.int64)+delta.astype(np.int64)
    if np.any(restored<0) or np.any(restored>65535):
        raise ReversibleError('invalid recovered sensor range')
    result=restored.astype('<u2').reshape(packet.shape).copy()
    if hashlib.sha256(result.tobytes()).hexdigest()!=packet.source_uint16_sha256:
        raise ReversibleError('bit-exact source digest mismatch')
    return result,True
