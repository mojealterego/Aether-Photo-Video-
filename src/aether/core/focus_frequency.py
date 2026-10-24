"""Bounded CPU Depth/Bokeh and Gaussian Frequency Separation on linear RGB."""
from __future__ import annotations
import math
import numpy as np
from .optical import _rgb


def guided_filter_gray(guide: np.ndarray, signal: np.ndarray, *, radius: int = 8, regularization: float = 1e-3) -> np.ndarray:
    """Classic grayscale guided filter with reflected borders."""
    import cv2
    I=np.asarray(guide,dtype=np.float32); p=np.asarray(signal,dtype=np.float32)
    if I.ndim!=2 or p.shape!=I.shape or not np.isfinite(I).all() or not np.isfinite(p).all():
        raise ValueError("Guide and signal must be finite, same-sized 2D arrays")
    if not 1<=radius<=64 or not math.isfinite(regularization) or regularization<=0:
        raise ValueError("Invalid guided filter settings")
    k=(2*radius+1,2*radius+1)
    def avg(x):return cv2.boxFilter(x,-1,k,borderType=cv2.BORDER_REFLECT_101)
    mi=avg(I);mp=avg(p)
    covariance=avg(I*p)-mi*mp
    variance=np.maximum(0,avg(I*I)-mi*mi)
    a=covariance/(variance+regularization)
    b=mp-a*mi
    return np.clip(avg(a)*I+avg(b),0,1).astype("f4")


def focus_bokeh_mask(linear_rgb:np.ndarray, *, depth:np.ndarray|None=None, focus_depth:float=.5,
                     depth_tolerance:float=.12, edge_weight:float=.55, guide_radius:int=8,
                     feather_sigma:float=1.,highlight_dampening:float=.8)->np.ndarray:
    """Relative depth plus Laplacian energy; not physical calibrated depth of field."""
    import cv2
    a=_rgb(linear_rgb)
    if not 0<=focus_depth<=1 or depth_tolerance<=0 or not 0<=edge_weight<=1 or feather_sigma<0 or not 0<=highlight_dampening<=1:
        raise ValueError("Invalid focus configuration")
    lum=(.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2]).astype("f4")
    detail=cv2.GaussianBlur(np.abs(cv2.Laplacian(lum,cv2.CV_32F,ksize=3)),(0,0),1.)
    energy=np.clip(detail/max(float(np.percentile(detail,98)),1e-6),0,1)
    if depth is not None:
        d=np.asarray(depth,dtype="f4")
        if d.shape!=lum.shape or not np.isfinite(d).all() or (d<0).any() or (d>1).any():
            raise ValueError("Invalid registered relative depth")
        geometry=np.exp(-.5*np.square((d-focus_depth)/depth_tolerance))
        result=((1-edge_weight)+edge_weight*energy)*geometry
    else:
        result=energy
    if highlight_dampening:
        result*=1-highlight_dampening*np.clip((lum-.8)/.2,0,1)*(1-energy)
    result=guided_filter_gray(lum,result.astype("f4"),radius=guide_radius)
    if feather_sigma:
        result=cv2.GaussianBlur(result,(0,0),sigmaX=feather_sigma)
    return np.clip(result,0,1).astype("f4")


def adaptive_master_base(optical_linear:np.ndarray,candidate_linear:np.ndarray,
                         inpaint_mask:np.ndarray,protected_mask:np.ndarray|None=None)->np.ndarray:
    """Only accepted and explicitly masked hallucinated details change source."""
    base=_rgb(optical_linear);guess=_rgb(candidate_linear)
    if guess.shape!=base.shape:raise ValueError("Candidate shape mismatch")
    m=np.asarray(inpaint_mask,dtype="f4")
    if m.shape!=base.shape[:2] or not np.isfinite(m).all() or (m<0).any() or (m>1).any():
        raise ValueError("Invalid damage mask")
    if protected_mask is not None:
        keep=np.asarray(protected_mask,dtype="f4")
        if keep.shape!=m.shape or not np.isfinite(keep).all() or (keep<0).any() or (keep>1).any():
            raise ValueError("Invalid protected mask")
        m=m*(1-keep)
    return (base*(1-m[:,:,None])+guess*m[:,:,None]).astype("f4")


def faithful_frequency_fusion(master_linear:np.ndarray,sr_linear:np.ndarray,
                              focus_mask:np.ndarray,*,sigma:float=9.,high_gain:float=1.,
                              protected_mask:np.ndarray|None=None)->np.ndarray:
    """Gaussian low band of MASTER with focus-protected high SR band."""
    import cv2
    base=_rgb(master_linear);sr=_rgb(sr_linear)
    if sr.shape!=base.shape:raise ValueError("SR and master must be registered")
    f=np.asarray(focus_mask,dtype="f4")
    if f.shape!=base.shape[:2] or not np.isfinite(f).all() or (f<0).any() or (f>1).any():
        raise ValueError("Invalid focus mask")
    if not math.isfinite(sigma) or sigma<=0 or not math.isfinite(high_gain) or not 0<=high_gain<=3:
        raise ValueError("Invalid frequency parameters")
    if protected_mask is not None:
        p=np.asarray(protected_mask,dtype="f4")
        if p.shape!=f.shape or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
            raise ValueError("Invalid protection mask")
        f=f*(1-p)
    low_base=cv2.GaussianBlur(base,(0,0),sigmaX=sigma)
    high_base=base-low_base
    high_sr=sr-cv2.GaussianBlur(sr,(0,0),sigmaX=sigma)
    return np.clip(low_base+(1-f[:,:,None])*high_base+f[:,:,None]*high_gain*high_sr,0,1).astype("f4")
