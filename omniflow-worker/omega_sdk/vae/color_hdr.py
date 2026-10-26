"""Aetheris Ω-VAE HDR color math (explicit reference matrices; no gamut clipping).

The matrices are fixed 3x3 approximations from SYS-AETHERIS-VISION-VAE-017;
not an OCIO color-managed production substitute without validation.
"""
from __future__ import annotations
import numpy as np
from .contracts import ColorContract, ContractError

_M=np.array([[.613097,.339523,.047379],
             [.070194,.916354,.013452],
             [.020616,.109570,.869815]],dtype=np.float64)
_INV=np.linalg.inv(_M)

def _scene(x):
    a=np.asarray(x)
    if a.dtype!=np.float32 or a.ndim!=4 or a.shape[1]!=3 or not np.isfinite(a).all():
        raise ContractError('finite float32 NCHW scene-linear RGB required')
    return a

def srgb_linear_to_acescg(x):
    a=_scene(x)
    return np.einsum('dc,bchw->bdhw',_M,a.astype(np.float64)).astype(np.float32)

def acescg_to_srgb_linear(x):
    a=_scene(x)
    return np.einsum('dc,bchw->bdhw',_INV,a.astype(np.float64)).astype(np.float32)

def signed_log1p(x,contract:ColorContract):
    contract.digest()
    a=_scene(x)
    scaled=a.astype(np.float64)*contract.exposure_scale
    return (np.sign(scaled)*np.log1p(np.abs(scaled)/contract.luminance_y0)).astype(np.float32)

def inverse_signed_log1p(x,contract:ColorContract):
    contract.digest()
    a=_scene(x)
    if float(np.max(np.abs(a)))>20:
        raise ContractError('HDR inversion would exceed safe numeric range')
    out=np.sign(a.astype(np.float64))*contract.luminance_y0*np.expm1(np.abs(a.astype(np.float64)))
    return (out/contract.exposure_scale).astype(np.float32)

def premultiply_alpha(rgb,alpha):
    a=_scene(rgb)
    al=np.asarray(alpha)
    if al.dtype!=np.float32 or al.shape!=(a.shape[0],1,a.shape[2],a.shape[3]) or not np.isfinite(al).all() or np.any((al<0)|(al>1)):
        raise ContractError('finite N1HW alpha in [0,1] required')
    return a*al

def unpremultiply_alpha(rgb,alpha,eps=1e-6):
    a=_scene(rgb)
    al=np.asarray(alpha)
    if al.dtype!=np.float32 or al.shape!=(a.shape[0],1,a.shape[2],a.shape[3]) or not np.isfinite(al).all() or np.any((al<0)|(al>1)):
        raise ContractError('finite N1HW alpha in [0,1] required')
    result=np.zeros_like(a)
    np.divide(a,al,out=result,where=al>eps)
    return result
