"""AETHERIS P028/P029: bounded native scene-linear optical correction.

These are deterministic optical estimates, not recovered documentary evidence.
Tile halos and cross-tile alignment remain the caller's responsibility.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve


def _rgb(image:np.ndarray,max_roi_pixels:int)->np.ndarray:
    data=np.asarray(image)
    if (data.ndim!=3 or data.shape[2]!=3 or data.dtype!=np.float32
            or data.shape[0]*data.shape[1]>max_roi_pixels
            or not np.isfinite(data).all()):
        raise ValueError("Finite scene-linear float32 RGB ROI within memory bound required")
    return data


def chromatic_aberration_correct_f32(
    image:np.ndarray,red_coeffs:tuple[float,float,float],
    blue_coeffs:tuple[float,float,float],
    center_xy:tuple[float,float]|None=None,
    max_roi_pixels:int=4_194_304,
)->np.ndarray:
    source=_rgb(image,max_roi_pixels)
    coeff=np.asarray([red_coeffs,blue_coeffs],dtype=np.float64)
    if coeff.shape!=(2,3) or not np.isfinite(coeff).all() or np.max(np.abs(coeff))>1:
        raise ValueError("Six finite bounded radial polynomial coefficients required")
    h,w=source.shape[:2]
    center=((w-1)/2,(h-1)/2) if center_xy is None else center_xy
    cx,cy=map(float,center)
    if not np.isfinite([cx,cy]).all():
        raise ValueError("Invalid optical center")
    if not coeff.any():
        return source.copy()
    yy,xx=np.meshgrid(np.arange(h,dtype=np.float32),
                      np.arange(w,dtype=np.float32),indexing="ij")
    norm=math.hypot(max(cx,w-cx),max(cy,h-cy))+1e-8
    dx=(xx-cx)/norm
    dy=(yy-cy)/norm
    r2=dx*dx+dy*dy
    out=source.copy()
    for channel,series in ((0,coeff[0]),(2,coeff[1])):
        scale=1+series[0]*r2+series[1]*r2*r2+series[2]*r2*r2*r2
        sample_x=cx+(xx-cx)*scale
        sample_y=cy+(yy-cy)*scale
        out[...,channel]=np.clip(map_coordinates(
            source[...,channel],[sample_y,sample_x],order=3,mode="reflect"),0,1)
    return out


def richardson_lucy_regularized_f32(
    image:np.ndarray,psf_kernel:np.ndarray,*,
    iterations:int=12,tv_lambda:float=0.0015,
    epsilon:float=1e-7,max_roi_pixels:int=4_194_304
)->np.ndarray:
    source=_rgb(image,max_roi_pixels)
    kernel=np.asarray(psf_kernel,dtype=np.float32)
    if (kernel.ndim!=2 or not all(d%2 and d>0 for d in kernel.shape)
            or max(kernel.shape)>127 or not np.isfinite(kernel).all()
            or np.min(kernel)<0 or float(kernel.sum())<=0):
        raise ValueError("PSF must be an odd-sized finite nonnegative 2D kernel")
    if (not isinstance(iterations,int) or not 1<=iterations<=50
            or not math.isfinite(tv_lambda) or not 0<=tv_lambda<=.1
            or not math.isfinite(epsilon) or not 0<epsilon<1e-3):
        raise ValueError("Invalid Richardson-Lucy regularization parameters")
    psf=kernel/np.sum(kernel)
    mirror=psf[::-1,::-1]
    pad_y,pad_x=kernel.shape[0]//2,kernel.shape[1]//2
    result=np.empty_like(source)
    for ch in range(3):
        plane=np.pad(np.clip(source[...,ch],0,1),
                     ((pad_y,pad_y),(pad_x,pad_x)),mode="reflect")
        estimate=np.maximum(plane,epsilon)
        for _ in range(iterations):
            reblurred=fftconvolve(estimate,psf,mode="same").astype(np.float32)
            ratio=plane/np.maximum(reblurred,epsilon)
            correction=fftconvolve(ratio,mirror,mode="same").astype(np.float32)
            if tv_lambda:
                gy,gx=np.gradient(estimate)
                den=np.sqrt(gx*gx+gy*gy)+1e-6
                nx,ny=gx/den,gy/den
                div=np.gradient(nx,axis=1)+np.gradient(ny,axis=0)
                tv=np.clip(1-tv_lambda*div,.5,2)
                estimate=(estimate/tv)*correction
            else:
                estimate=estimate*correction
            estimate=np.clip(estimate,0,1)
        result[...,ch]=estimate[pad_y:pad_y+source.shape[0],
                              pad_x:pad_x+source.shape[1]]
    return result
