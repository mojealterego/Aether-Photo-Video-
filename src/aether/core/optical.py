"""Bounded linear-light optical operations for Galaxy S24 Ultra CPU backend.

Single-patch, float32 RGB LINEAR LIGHT; caller supplies halo and global coords.
No GPU, CUDA, PyTorch, ComfyUI or full-image allocation.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def _rgb(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image, dtype=np.float32)
    if a.ndim != 3 or a.shape[2] != 3 or not np.isfinite(a).all():
        raise ValueError("Expected finite (H,W,3) float32 RGB")
    return a


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    x = _rgb(image)
    if x.min() < 0 or x.max() > 1:
        raise ValueError("Expected normalized sRGB [0,1]")
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4).astype("f4")


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    x = np.clip(_rgb(image), 0, 1)
    return np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055).astype("f4")


@dataclass(frozen=True)
class LCAChannel:
    radial: tuple[float, float, float, float] = (0., 0., 0., 0.)
    tangential: tuple[float, float] = (0., 0.)

    def __post_init__(self):
        v = (*self.radial, *self.tangential)
        if len(self.radial) != 4 or len(self.tangential) != 2 or not all(math.isfinite(x) and abs(x) <= .05 for x in v):
            raise ValueError("Invalid LCA coefficients")


def correct_lca(tile: np.ndarray, *, origin_xy: tuple[int, int], full_wh: tuple[int, int],
                red: LCAChannel = LCAChannel(), blue: LCAChannel = LCAChannel()) -> np.ndarray:
    """8th-degree radial+tangential LCA, green reference, global tile coords.

    Caller must supply a halo covering the maximum chromatic displacement.
    """
    import cv2
    x = _rgb(tile)
    w, h = full_wh
    x0, y0 = origin_xy
    if min(w, h) < 2 or x0 < 0 or y0 < 0 or x0 + x.shape[1] > w or y0 + x.shape[0] > h:
        raise ValueError("Invalid global tile geometry")
    if red == LCAChannel() and blue == LCAChannel():
        return x.copy()
    yy, xx = np.indices(x.shape[:2], dtype=np.float32)
    xx += x0
    yy += y0
    cx, cy = (w-1)/2, (h-1)/2
    norm = max(1., math.hypot(cx, cy))
    dx, dy = (xx-cx)/norm, (yy-cy)/norm
    r2 = dx*dx + dy*dy
    out = x.copy()
    for channel, model in ((0, red), (2, blue)):
        if model == LCAChannel():
            continue
        a2, a4, a6, a8 = model.radial
        p1, p2 = model.tangential
        scale = 1 + r2*(a2 + r2*(a4 + r2*(a6 + r2*a8)))
        mx = (cx + norm*(dx*scale + 2*p1*dx*dy + p2*(r2+2*dx*dx)) - x0).astype("f4")
        my = (cy + norm*(dy*scale + p1*(r2+2*dy*dy) + 2*p2*dx*dy) - y0).astype("f4")
        out[:,:,channel] = cv2.remap(x[:,:,channel], mx, my,
                                    interpolation=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REFLECT_101)
    return out


def anisotropic_gaussian_psf(sigma_radial: float, sigma_tangential: float,
                             angle_radians: float, radius: int = 9) -> np.ndarray:
    """Anisotropic Gaussian PSF approximation; not a measured optical kernel."""
    if not all(math.isfinite(v) for v in (sigma_radial,sigma_tangential,angle_radians)) or min(sigma_radial,sigma_tangential) <= 0 or not 1 <= radius <= 64:
        raise ValueError("Invalid PSF")
    yy, xx = np.mgrid[-radius:radius+1,-radius:radius+1].astype("f4")
    c,s = math.cos(angle_radians), math.sin(angle_radians)
    xr,yr = c*xx+s*yy, -s*xx+c*yy
    k = np.exp(-.5*((xr/sigma_radial)**2+(yr/sigma_tangential)**2))
    return (k/k.sum()).astype("f4")


def _tv_divergence(x: np.ndarray) -> np.ndarray:
    """Neumann boundary TV divergence."""
    dx = np.zeros_like(x)
    dy = np.zeros_like(x)
    dx[:,:-1] = x[:,1:] - x[:,:-1]
    dy[:-1] = x[1:] - x[:-1]
    g = np.sqrt(dx*dx+dy*dy+1e-8)
    dx /= g
    dy /= g
    div = dx + dy
    div[:,1:] -= dx[:,:-1]
    div[1:] -= dy[:-1]
    return div


def richardson_lucy_tv(image_linear: np.ndarray, psf: np.ndarray, *, iterations: int = 6,
                       tv_weight: float = .001, focus: np.ndarray | None = None,
                       epsilon: float = 1e-6, ratio_cap: float = 4.) -> np.ndarray:
    """RL-TV with explicit A-transpose-ones for zero Dirichlet boundary model.

    Border values may deviate with zero-boundary physics. Application must
    provide a reflect-extended halo and crop it after filtering; do not run
    with an insufficient halo and assume seam-free output.
    """
    import cv2
    y = _rgb(image_linear)
    if y.min() < 0 or y.max() > 1 or not 0 <= iterations <= 30 or not 0 <= tv_weight <= .1 or epsilon <= 0 or ratio_cap <= 1:
        raise ValueError("Invalid RL arguments")
    k = np.asarray(psf,dtype="f4")
    if k.ndim != 2 or k.shape[0]%2 == 0 or k.shape[1]%2 == 0 or min(k.shape) < 1 or not np.isfinite(k).all() or (k < 0).any() or k.sum() <= 0:
        raise ValueError("PSF must be an odd, normalized nonnegative matrix")
    k = k/k.sum()
    gate = None
    if focus is not None:
        gate = np.asarray(focus,dtype="f4")
        if gate.shape != y.shape[:2] or not np.isfinite(gate).all() or (gate < 0).any() or (gate > 1).any():
            raise ValueError("Invalid focus mask")
        if not np.any(gate):
            return y.copy()
    if iterations == 0 or k.shape == (1,1):
        return y.copy()
    kt = np.flip(k,axis=(0,1)).copy()
    def conv(a,ker):
        return cv2.filter2D(a,cv2.CV_32F,ker,borderType=cv2.BORDER_CONSTANT)
    sensitivity = np.maximum(conv(np.ones(y.shape[:2],dtype="f4"),kt),epsilon)
    result = np.maximum(y.copy(),epsilon)
    for _ in range(iterations):
        for c in range(3):
            x = result[:,:,c]
            blur = np.maximum(conv(x,k),epsilon)
            ratio = np.clip(y[:,:,c]/blur,0,ratio_cap)
            correction = conv(ratio,kt)
            denom = sensitivity-tv_weight*_tv_divergence(x)
            candidate = np.clip(x*correction/np.maximum(denom,.1),0,1)
            result[:,:,c] = candidate if gate is None else x+(candidate-x)*gate
    return result
