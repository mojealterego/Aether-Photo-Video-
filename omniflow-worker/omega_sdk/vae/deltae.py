"""CIEDE2000 (CIE 142 / Sharma 2005) on explicitly supplied CIE L*a*b*.

Requires color-managed D65 Lab upstream. This function must not be applied
on raw ACEScg RGB triplets, which are NOT Lab and use D60 white point.
"""
from __future__ import annotations
import numpy as np

class ColorMetricError(ValueError):
    pass

def ciede2000(lab1: np.ndarray,lab2: np.ndarray)->np.ndarray:
    a=np.asarray(lab1,dtype=np.float64)
    b=np.asarray(lab2,dtype=np.float64)
    if a.shape!=b.shape or a.ndim<1 or a.shape[-1]!=3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ColorMetricError('matching finite CIE Lab arrays with last dimension 3 required')
    L1,a1,b1=np.moveaxis(a,-1,0)
    L2,a2,b2=np.moveaxis(b,-1,0)
    C1=np.hypot(a1,b1);C2=np.hypot(a2,b2)
    cbar=(C1+C2)/2
    p7=25.0**7
    g=.5*(1-np.sqrt(cbar**7/(cbar**7+p7)))
    ap1=(1+g)*a1;ap2=(1+g)*a2
    cp1=np.hypot(ap1,b1);cp2=np.hypot(ap2,b2)
    hp1=np.mod(np.degrees(np.arctan2(b1,ap1)),360.)
    hp2=np.mod(np.degrees(np.arctan2(b2,ap2)),360.)
    chromatic=(cp1*cp2)>1e-22
    dh=hp2-hp1
    dh=np.where(dh>180,dh-360,np.where(dh < -180,dh+360,dh))
    dh=np.where(chromatic,dh,0.)
    dL=L2-L1;dC=cp2-cp1
    dH=2*np.sqrt(cp1*cp2)*np.sin(np.radians(dh/2))
    hsum=hp1+hp2
    hdifference=np.abs(hp1-hp2)
    hbar=np.where(~chromatic,hsum,
                  np.where(hdifference<=180,hsum/2,
                           np.where(hsum<360,(hsum+360)/2,(hsum-360)/2)))
    lb=(L1+L2)/2
    cb=(cp1+cp2)/2
    T=(1-.17*np.cos(np.radians(hbar-30))
        +.24*np.cos(np.radians(2*hbar))
        +.32*np.cos(np.radians(3*hbar+6))
        -.20*np.cos(np.radians(4*hbar-63)))
    SL=1+.015*(lb-50)**2/np.sqrt(20+(lb-50)**2)
    SC=1+.045*cb
    SH=1+.015*cb*T
    dtheta=30*np.exp(-((hbar-275)/25)**2)
    RC=2*np.sqrt(cb**7/(cb**7+p7))
    RT=-np.sin(np.radians(2*dtheta))*RC
    lterm=dL/SL;cterm=dC/SC;hterm=dH/SH
    square=lterm*lterm+cterm*cterm+hterm*hterm+RT*cterm*hterm
    return np.sqrt(np.maximum(square,0.))
