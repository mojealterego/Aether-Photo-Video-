"""Empirical point-source PSF and radial lateral-CA calibration utilities.

The data MUST come from a suitably exposed, unsaturated point-source image
at the relevant focal length, aperture, focus distance and sensor position.
No generic calibration can recover the true optical PSF of every lens.
"""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np


def measure_point_source_psf(point_crop:np.ndarray)->np.ndarray:
    a=np.asarray(point_crop,dtype=np.float32)
    if (a.ndim!=2 or a.shape[0]!=a.shape[1] or a.shape[0]%2!=1 or
            a.shape[0]<3 or a.shape[0]>15 or not np.isfinite(a).all()):
        raise ValueError('Provide a finite odd square 3..15 crop around an unsaturated point source')
    if np.min(a)<0:raise ValueError('Point source cannot have negative pixel values')
    background=float(np.percentile(np.concatenate((a[0],a[-1],a[1:-1,0],a[1:-1,-1])),15))
    signal=np.maximum(a-background,0)
    if np.max(a)<=background+1e-5 or float(signal.sum())<1e-4:
        raise ValueError('No measurable light source above background')
    return (signal/signal.sum()).astype(np.float32)


def save_psf_calibration(path:Path,kernel:np.ndarray)->Path:
    p=Path(path)
    if p.suffix.lower()!='.npy':raise ValueError('PSF calibration must be an .npy array')
    kernel=validate_psf(kernel)
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    tmp=p.with_name(p.name+'.partial')
    try:
        with tmp.open('xb') as f:
            np.save(f,kernel,allow_pickle=False)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:tmp.unlink(missing_ok=True)
    return p


def validate_psf(kernel:np.ndarray)->np.ndarray:
    k=np.asarray(kernel,dtype=np.float32)
    if (k.ndim!=2 or k.shape[0]!=k.shape[1] or k.shape[0]%2!=1 or
        k.shape[0]<3 or k.shape[0]>15 or not np.isfinite(k).all() or np.min(k)<0 or
            float(k.sum())<=1e-8):
        raise ValueError('Invalid measured point spread function (odd square <=15)')
    return k / float(k.sum())


def calibrate_radial_ca(green_points:np.ndarray,observed_channel_points:np.ndarray,
                        sensor_size:tuple[int,int])->float:
    g=np.asarray(green_points,dtype=np.float64)
    o=np.asarray(observed_channel_points,dtype=np.float64)
    w,h=sensor_size
    if (g.shape!=o.shape or g.ndim!=2 or g.shape[1]!=2 or g.shape[0]<8
            or min(w,h)<16 or not np.isfinite(g).all() or not np.isfinite(o).all()):
        raise ValueError('At least 8 corresponding finite two-dimensional features needed')
    center=np.array([(w-1)*.5,(h-1)*.5])
    dx=g-center
    radius2=np.sum(dx*dx,axis=1)/max(float(np.sum(center*center)),1.)
    design=dx*radius2[:,None]
    denominator=float((design**2).sum())
    if denominator<1e-10:raise ValueError('Insufficient field coverage for CA calibration')
    k=float(np.sum((o-g)*design)/denominator)
    if abs(k)>.01:raise ValueError('Measured CA beyond supported radial model')
    return k