"""Bounded CPU algorithms for S24 Ultra: damage *proposals* and masked hypothesis fusion.

No identity is inferred or recovered. A neural inpaint result is a *hypothesis*,
not original sensor data. No PyTorch, OpenCV, network, or model weights required.
All floating-point color operations use linear-light sRGB primaries.
"""
from __future__ import annotations

from collections import deque
import math
import numpy as np



def _box_mean(value: np.ndarray, window: int) -> np.ndarray:
    """Reflect-padded local mean. Explicitly bounded to <=2MP previews by callers."""
    if window < 1 or window % 2 == 0:
        raise ValueError('Window size must be odd positive')
    r = window // 2
    padded = np.pad(value.astype(np.float32, copy=False), ((r, r), (r, r)), mode='reflect')
    # Double precision accumulation minimizes cancellation in variance estimates.
    ii = np.pad(np.cumsum(np.cumsum(padded, axis=0, dtype=np.float64),
                               axis=1, dtype=np.float64), ((1, 0), (1, 0)))
    sums = ii[window:, window:] - ii[:-window, window:] - ii[window:, :-window] + ii[:-window, :-window]
    return (sums / (window * window)).astype(np.float32)


def _components(binary: np.ndarray):
    """Yield bounding box + exact interior mask for 8-connected components.

    Scanning is restricted to small analysis previews (<=2MP), not full photos.
    """
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    for sy, sx in zip(*np.where(binary)):
        if visited[sy, sx]:
            continue
        q = deque([(int(sy), int(sx))])
        visited[sy, sx] = True
        area = 0
        x0 = x1 = int(sx)
        y0 = y1 = int(sy)
        while q:
            y, x = q.popleft()
            area += 1
            x0 = min(x0, x); x1 = max(x1, x)
            y0 = min(y0, y); y1 = max(y1, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        yield x0, y0, x1 + 1, y1 + 1, area


def propose_damage_mask(image: np.ndarray, *, region: np.ndarray | None = None,
                        feather: float = 2.0, flat_std: float = 0.018,
                        edge_threshold: float = 0.07) -> np.ndarray:
    """Conservative heuristic for solid covers and regular mosaic textures.

    Only a reviewable proposal; natural geometry can trigger false positives.
    The input must be a bounded <=2MP uint8 RGB or float32 RGB preview.
    A supplied ROI *only restricts* proposals, it never forces a modification.
    """
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 16:
        raise ValueError('RGB HWC preview >=16x16 required')
    h, w = rgb.shape[:2]
    if h * w > 2_097_152:
        raise ValueError('Detector operates only on <=2MP preview')
    if np.issubdtype(rgb.dtype, np.integer):
        rgb = rgb.astype(np.float32) / float(np.iinfo(rgb.dtype).max)
    else:
        rgb = rgb.astype(np.float32, copy=False)
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError('Expected finite RGB values in [0,1]')
    if region is None:
        roi = np.ones((h, w), dtype=bool)
    else:
        m = np.asarray(region)
        if m.shape != (h,w) or not np.isfinite(m).all():
            raise ValueError('Invalid ROI mask')
        roi = m > .5
    if not (math.isfinite(feather) and 0 <= feather <= 32):
        raise ValueError('Invalid feather radius')
    if not (0 < flat_std < .25 and 0 < edge_threshold < 1):
        raise ValueError('Invalid thresholds')
    gray = np.sum(rgb * np.array([.2126, .7152, .0722],dtype=np.float32),axis=2)
    mean3 = _box_mean(gray,3)
    std3 = np.sqrt(np.maximum(_box_mean(gray*gray,3)-mean3*mean3,0))
    mean7 = _box_mean(gray,7)
    std7 = np.sqrt(np.maximum(_box_mean(gray*gray,7)-mean7*mean7,0))
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edge = ((gx > edge_threshold) | (gy > edge_threshold)).astype(np.float32)
    flat_density = _box_mean((std3 < flat_std).astype(np.float32), 15)
    edge_density = _box_mean(edge, 15)
    # Repeated flat cells together with a *nonzero* grid-step density.
    mosaic = (flat_density > .14) & (edge_density > .075) & (edge_density < .62)
    result = mosaic.astype(np.float32)
    # A flat, finite rectangle surrounded by different colors can indicate a cover.
    solid_interior = (std7 < flat_std) & roi
    limit = max(35, int(h*w / 6000))
    for x0,y0,x1,y1,area in _components(solid_interior):
        bw,bh=x1-x0,y1-y0
        if (area < limit or bw < 9 or bh < 9 or area > h*w*.6 or
            x0==0 or y0==0 or x1==w or y1==h or area/(bw*bh) < .72):
            continue
        pad = 7
        a0,b0=max(0,x0-pad),max(0,y0-pad)
        a1,b1=min(w,x1+pad),min(h,y1+pad)
        ring = gray[b0:b1,a0:a1].copy()
        interior = np.zeros_like(ring,dtype=bool)
        interior[y0-b0:y1-b0,x0-a0:x1-a0] = True
        if (~interior).sum() < 10:
            continue
        core_gray = gray[y0:y1,x0:x1]
        if abs(float(np.median(core_gray)-np.median(ring[~interior]))) < .085:
            continue
        result[max(0,y0-3):min(h,y1+3),max(0,x0-3):min(w,x1+3)] = 1
    # Smooth only the proposal, clamp to explicit region *after* feathering.
    if feather > 0:
        from PIL import Image, ImageFilter
        converted = Image.fromarray(np.uint8(np.clip(result,0,1)*255),'L')
        radius=float(feather)
        result = np.asarray(converted.filter(ImageFilter.GaussianBlur(radius)),dtype=np.float32)/255
    return np.clip(result * roi.astype(np.float32),0,1).astype(np.float32)


def anisotropic_deblock(image: np.ndarray, mask: np.ndarray, *, iterations: int=10,
                         kappa: float=.1, step: float=.18) -> np.ndarray:
    """Perona–Malik edge-sensitive diffusion (small tile only); NOT inpainting."""
    x = np.asarray(image,dtype=np.float32)
    m = np.asarray(mask,dtype=np.float32)
    if (x.ndim!=3 or x.shape[2]!=3 or m.shape!=x.shape[:2] or
        not np.isfinite(x).all() or not np.isfinite(m).all() or
        x.min()<0 or x.max()>1 or m.min()<0 or m.max()>1):
        raise ValueError('Bad RGB tile/mask')
    if not (0 <= iterations <= 60 and 0 < kappa <= 1 and 0 < step <= .25):
        raise ValueError('Diffusion stability contract violated')
    if not iterations or m.max() <= 0: return x.copy()
    u=x.copy()
    for _ in range(iterations):
        p=np.pad(u,((1,1),(1,1),(0,0)),mode='reflect')
        diffs=(p[:-2,1:-1]-u,p[2:,1:-1]-u,
               p[1:-1,:-2]-u,p[1:-1,2:]-u)
        adjustment=np.zeros_like(u)
        for d in diffs:
            conduction=np.exp(-np.sum(d*d,axis=2,keepdims=True)/(kappa*kappa))
            adjustment += conduction*d
        u += step*adjustment
        np.clip(u,0,1,out=u)
    return x*(1-m[...,None])+u*m[...,None]


# CIE 1976 LAB D65, with *linear* sRGB primaries and float32 arithmetic.
_TO_XYZ = np.array([[.4124564,.3575761,.1804375],
                    [.2126729,.7151522,.0721750],
                    [.0193339,.1191920,.9503041]],dtype=np.float32)
_FROM_XYZ = np.linalg.inv(_TO_XYZ).astype(np.float32)
_WHITE = np.array([.95047,1.,1.08883],dtype=np.float32)


def _lab(linear_rgb: np.ndarray) -> np.ndarray:
    xyz = (np.asarray(linear_rgb,dtype=np.float32) @ _TO_XYZ.T) / _WHITE
    delta = 6/29
    f = np.where(xyz>delta**3, np.cbrt(np.maximum(xyz,0)), xyz/(3*delta**2)+4/29)
    return np.stack((116*f[...,1]-16, 500*(f[...,0]-f[...,1]),
                     200*(f[...,1]-f[...,2])),axis=-1).astype(np.float32)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    fy=(lab[...,0]+16)/116
    fx=fy+lab[...,1]/500
    fz=fy-lab[...,2]/200
    delta=6/29
    f=np.stack((fx,fy,fz),axis=-1)
    xyz=np.where(f>delta,f**3,3*delta**2*(f-4/29))*_WHITE
    return np.clip(xyz @ _FROM_XYZ.T,0,1).astype(np.float32)


def _color_match(ref: np.ndarray,cand: np.ndarray,mask: np.ndarray,strength:float)->np.ndarray:
    core=mask>.85
    if np.count_nonzero(core)<32: return cand
    ring=(_box_mean(mask,15)>.01)&(mask<.01)
    if np.count_nonzero(ring)<32:return cand
    a=_lab(ref);b=_lab(cand)
    mu_a=np.mean(a[ring],axis=0); sd_a=np.std(a[ring],axis=0)
    mu_b=np.mean(b[core],axis=0); sd_b=np.std(b[core],axis=0)
    ratio=np.clip((sd_a+.001)/(sd_b+.001),.75,1.25)
    transferred=np.clip((b-mu_b)*ratio+mu_a,
                        np.array([0,-128,-128],dtype=np.float32),
                        np.array([100,127,127],dtype=np.float32))
    return np.clip(cand*(1-strength)+_from_lab(transferred)*strength,0,1)


def masked_linear_fusion(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray,
                         *, protected: np.ndarray|None=None, strength:float=1.,
                         color_match_strength:float=0.) -> np.ndarray:
    """Adaptive *low-frequency* base replacement only where permitted by the mask.

    Outside the mask, original float32 samples are unchanged. LAB adaptation is
    optional because arbitrary neighboring material need not be reference skin.
    """
    ref=np.asarray(reference,dtype=np.float32)
    cand=np.asarray(candidate,dtype=np.float32)
    m=np.asarray(mask,dtype=np.float32)
    if (ref.ndim!=3 or ref.shape[-1]!=3 or cand.shape!=ref.shape or
        m.shape!=ref.shape[:2] or not all(np.isfinite(v).all() for v in (ref,cand,m)) or
        np.min(ref)<0 or np.max(ref)>1 or np.min(cand)<0 or np.max(cand)>1 or
        np.min(m)<0 or np.max(m)>1):
        raise ValueError('Invalid normalized RGB reference, hypothesis or mask')
    if not (math.isfinite(strength) and 0<=strength<=1 and
            math.isfinite(color_match_strength) and 0<=color_match_strength<=1):
        raise ValueError('Invalid fusion strengths')
    if protected is not None:
        p=np.asarray(protected,dtype=np.float32)
        if p.shape!=m.shape or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
            raise ValueError('Invalid protected mask')
        m=m*(1-p)
    alpha=m*strength
    if float(alpha.max())==0:return ref.copy()
    if color_match_strength>0:
        cand=_color_match(ref,cand,m,color_match_strength)
    return np.clip(ref+(cand-ref)*alpha[...,None],0,1).astype(np.float32)