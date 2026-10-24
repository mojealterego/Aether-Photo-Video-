"""Multilevel invertible Haar/DWT fusion for individual LINEAR float32 RGB tiles.
No full-frame 400 MP allocation; caller is responsible for tile halo and WOLA.
Generated high-frequency content is not evidence of recovered original detail.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class HaarBands:
    horizontal: np.ndarray
    vertical: np.ndarray
    diagonal: np.ndarray
    original_shape: tuple[int,int,int]

def _rgb(a):
    a=np.asarray(a,dtype=np.float32)
    if a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<1 or not np.isfinite(a).all():
        raise ValueError("Expected finite float32 RGB H,W,3")
    return a

def _haar_step(a):
    h,w,_=a.shape
    if h%2 or w%2:
        a=np.pad(a,((0,h%2),(0,w%2),(0,0)),mode="edge")
    p,q,r,s=a[0::2,0::2],a[0::2,1::2],a[1::2,0::2],a[1::2,1::2]
    return ((p+q+r+s)*.25,
            HaarBands((p+q-r-s)*.25,(p-q+r-s)*.25,(p-q-r+s)*.25,(h,w,3)))

def haar_decompose(image:np.ndarray,*,levels:int=3):
    if type(levels) is not int or not 1<=levels<=10:
        raise ValueError("Haar levels must be 1..10")
    current=_rgb(image)
    bands=[]
    for _ in range(levels):
        current,b=_haar_step(current)
        bands.append(b)
    return current,bands

def haar_reconstruct(low:np.ndarray,bands:list[HaarBands])->np.ndarray:
    current=_rgb(low)
    if not bands:
        raise ValueError("Expected at least one Haar level")
    for b in reversed(bands):
        h,w,c=b.original_shape
        shape=((h+1)//2,(w+1)//2,c)
        if current.shape!=shape or any(x.shape!=shape for x in (b.horizontal,b.vertical,b.diagonal)):
            raise ValueError("Mismatched wavelet pyramid")
        hp,vp,dp=b.horizontal,b.vertical,b.diagonal
        out=np.empty((shape[0]*2,shape[1]*2,c),np.float32)
        out[0::2,0::2]=current+hp+vp+dp
        out[0::2,1::2]=current+hp-vp-dp
        out[1::2,0::2]=current-hp+vp-dp
        out[1::2,1::2]=current-hp-vp+dp
        current=out[:h,:w]
    return current

def _down(mask):
    h,w=mask.shape
    m=np.pad(mask,((0,h%2),(0,w%2)),mode="edge")
    return (m[0::2,0::2]+m[0::2,1::2]+m[1::2,0::2]+m[1::2,1::2])*.25

def wavelet_faithful_fusion(master_linear:np.ndarray,sr_linear:np.ndarray,
                            focus_mask:np.ndarray,*,levels:int=3,high_gain:float=1.,
                            protected_mask:np.ndarray|None=None)->np.ndarray:
    """Preserve master's coarse band; focus-selectively inject SR detail bands.

    Protected regions are replaced by source after reconstruction. The caller
    must provide halo/feathering for spatial continuity between tiles.
    """
    base=_rgb(master_linear)
    sr=_rgb(sr_linear)
    if sr.shape!=base.shape:
        raise ValueError("SR/master shape mismatch")
    mask=np.asarray(focus_mask,dtype=np.float32)
    if mask.shape!=base.shape[:2] or not np.isfinite(mask).all() or (mask<0).any() or (mask>1).any():
        raise ValueError("Invalid focus mask")
    if not math.isfinite(high_gain) or not 0<=high_gain<=2:
        raise ValueError("high_gain outside 0..2")
    if protected_mask is not None:
        protected=np.asarray(protected_mask,dtype=np.float32)
        if protected.shape!=mask.shape or not np.isfinite(protected).all() or (protected<0).any() or (protected>1).any():
            raise ValueError("Invalid protection mask")
        mask=mask*(1-protected)
    if not np.any(mask) or high_gain==0:
        return base.copy()
    low,bands=haar_decompose(base,levels=levels)
    _,sr_bands=haar_decompose(sr,levels=levels)
    output=[]
    for orig,new in zip(bands,sr_bands):
        mask=_down(mask)
        m=mask[:,:,None]
        output.append(HaarBands(
            orig.horizontal*(1-m)+new.horizontal*(m*high_gain),
            orig.vertical*(1-m)+new.vertical*(m*high_gain),
            orig.diagonal*(1-m)+new.diagonal*(m*high_gain),
            orig.original_shape))
    fused=haar_reconstruct(low,output)
    if protected_mask is not None:
        fused=base*protected[:,:,None]+fused*(1-protected[:,:,None])
    return np.clip(fused,0.,1.).astype(np.float32)
