"""AETHERIS native photographic operators — SYS-AETHERIS-OMNIFLOW-016.

Explicit linear-sRGB to display-sRGB transfer. Disk-backed overlap
accumulation avoids allocating a full-frame float32 RGB GPU tensor.
This does NOT itself implement 400 MP upscaling or validate model inference.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np
import tifffile
from PIL import ImageCms

from .protocol import (
    ExecutionContext, NodeExecutionMode, NodeManifest,
    OmegaNode, TrustClass, OMEGA_ABI_VERSION
)


def linear_to_srgb_u16(linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 display sRGB transfer from linear-sRGB float32."""
    a=np.asarray(linear,dtype=np.float32)
    if a.ndim!=3 or a.shape[2]!=3 or not np.isfinite(a).all():
        raise ValueError("Finite HxWx3 linear RGB required")
    v=np.clip(a,0,1)
    out=np.where(v<=0.0031308,12.92*v,1.055*np.power(v,1/2.4)-0.055)
    return np.rint(np.clip(out,0,1)*65535).astype(np.uint16)


class DiskBackedStitcher:
    """Weighted overlap compositor using disk-backed float32 accumulation.

    Peak file-backed scratch: H*W*16 bytes, output: H*W*6 bytes.
    The caller owns the work directory lifecycle; no original is overwritten.
    """
    def __init__(self,height:int,width:int,scratch:Path,*,reserve_bytes:int=1<<30):
        if min(height,width)<1 or height*width>420_000_000:
            raise ValueError("Unsupported image dimensions")
        self.height,self.width=int(height),int(width)
        self.scratch=Path(scratch)
        if self.scratch.exists():
            raise FileExistsError("Scratch directory must be new to prevent stale state")
        self.scratch.mkdir(parents=True,mode=0o700)
        if shutil.disk_usage(self.scratch).free<height*width*22+reserve_bytes:
            raise OSError("Insufficient disk space for out-of-core reconstruction")
        self.pixels=np.memmap(self.scratch/"pixels.f32",dtype="f4",mode="w+",shape=(height,width,3))
        self.weights=np.memmap(self.scratch/"weights.f32",dtype="f4",mode="w+",shape=(height,width))
        self._closed=False
        self.tiles_written=0

    def add_tile(self,y:int,x:int,linear_rgb:np.ndarray,window:np.ndarray|None=None)->None:
        if self._closed:
            raise RuntimeError("Stitcher is closed")
        pixels=np.asarray(linear_rgb,dtype=np.float32)
        if pixels.ndim!=3 or pixels.shape[2]!=3 or not np.isfinite(pixels).all():
            raise ValueError("Tile must be finite HxWx3 float RGB")
        h,w=pixels.shape[:2]
        if h<1 or w<1 or y<0 or x<0 or y+h>self.height or x+w>self.width:
            raise ValueError("Tile outside target image")
        if window is None:
            wy=np.hanning(h+2)[1:-1].astype(np.float32)
            wx=np.hanning(w+2)[1:-1].astype(np.float32)
            win=np.maximum(np.outer(wy,wx),1e-4)
        else:
            win=np.asarray(window,dtype=np.float32)
            if win.shape!=(h,w) or not np.isfinite(win).all() or np.any(win<=0):
                raise ValueError("Positive finite weight window required")
        self.pixels[y:y+h,x:x+w]+=pixels*win[...,None]
        self.weights[y:y+h,x:x+w]+=win
        self.tiles_written+=1

    def write_srgb16_bigtiff(self,output:Path,*,tile_size:int=256,
                             source_sha256:str|None=None)->dict[str,Any]:
        output=Path(output)
        if output.exists():
            raise FileExistsError(output)
        if tile_size<16 or tile_size%16:
            raise ValueError("TIFF tile size must be a multiple of 16")
        if not self.tiles_written:
            raise ValueError("ERR_TILE_COVERAGE_GAP: no tiles")
        for y in range(0,self.height,tile_size):
            if np.any(self.weights[y:y+tile_size,:]<=0):
                raise ValueError("ERR_TILE_COVERAGE_GAP: uncovered pixels")
        self.pixels.flush()
        self.weights.flush()
        output.parent.mkdir(parents=True,exist_ok=True)
        icc=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        provenance=json.dumps({
            "engine":"AETHERIS_OMNIFLOW_OMEGA_v1.1",
            "source_sha256":source_sha256,
            "tile_count":self.tiles_written,
            "output_color_space":"display-sRGB",
            "output_transfer":"IEC61966-2-1",
            "bit_depth":16},sort_keys=True)
        temp=output.with_name("."+output.name+"."+uuid.uuid4().hex+".partial")
        def tiles():
            for y in range(0,self.height,tile_size):
                for x in range(0,self.width,tile_size):
                    h=min(tile_size,self.height-y)
                    w=min(tile_size,self.width-x)
                    accumulated=np.asarray(self.pixels[y:y+h,x:x+w],dtype=np.float32)
                    weights=np.asarray(self.weights[y:y+h,x:x+w],dtype=np.float32)
                    rgb=linear_to_srgb_u16(accumulated/weights[...,None])
                    padded=np.zeros((tile_size,tile_size,3),dtype=np.uint16)
                    padded[:h,:w,:]=rgb
                    yield padded
        try:
            with tifffile.TiffWriter(temp,bigtiff=True) as writer:
                writer.write(tiles(),shape=(self.height,self.width,3),
                             dtype=np.uint16,photometric="rgb",
                             tile=(tile_size,tile_size),metadata=None,
                             description=provenance,
                             extratags=[(34675,"B",len(icc),icc,True)])
            with tifffile.TiffFile(temp) as tif:
                p=tif.pages[0]
                if not tif.is_bigtiff or p.shape!=(self.height,self.width,3) or p.dtype!=np.uint16:
                    raise ValueError("BigTIFF verification failed")
                if "InterColorProfile" not in p.tags:
                    raise ValueError("Missing ICC profile")
            # Exclusive publication; os.replace would silently overwrite an existing master.
            os.link(temp,output)
            digest=hashlib.sha256()
            with output.open("rb") as file:
                for chunk in iter(lambda:file.read(1<<20),b""):
                    digest.update(chunk)
            return {
                "output":str(output),"sha256":"sha256:"+digest.hexdigest(),
                "dimensions":[self.height,self.width,3],"bit_depth":16,
                "color_space":"display-sRGB","tiles_written":self.tiles_written
            }
        finally:
            temp.unlink(missing_ok=True)

    def close(self):
        if not self._closed:
            self.pixels.flush()
            self.weights.flush()
            del self.pixels,self.weights
            self._closed=True

    def __enter__(self): return self
    def __exit__(self,*_): self.close()


class I002_LoadNEF(OmegaNode):
    """Nikon NEF -> bounded linear-sRGB float32; no modification of source."""
    @classmethod
    def manifest(cls)->NodeManifest:
        return NodeManifest(
            "I002_LoadNEF","1.1.0","SOURCE_AND_COLOR",OMEGA_ABI_VERSION,
            NodeExecutionMode.PURE_DETERMINISTIC,TrustClass.T0_CORE_TRUSTED,(),False)

    def execute(self,ctx:ExecutionContext,inputs:dict[str,Any],
                config:dict[str,Any])->dict[str,Any]:
        source=Path(inputs["nef_path"])
        if (source.suffix.lower()!=".nef" or not source.is_file()
                or source.stat().st_size>int(config.get("max_file_bytes",256<<20))):
            raise ValueError("Invalid or oversized NEF input")
        with source.open("rb") as file:
            if file.read(4) not in (b"II*\x00",b"MM\x00*"):
                raise ValueError("Invalid NEF TIFF header")
        digest=hashlib.sha256()
        with source.open("rb") as file:
            for chunk in iter(lambda:file.read(1<<20),b""):
                digest.update(chunk)
        source_sha="sha256:"+digest.hexdigest()
        if inputs.get("expected_sha256") not in (None,source_sha):
            raise ValueError("NEF checksum mismatch")
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError("rawpy/LibRaw required for real NEF decoding") from exc
        with rawpy.imread(str(source)) as raw:
            w,h=int(raw.sizes.width),int(raw.sizes.height)
            if min(w,h)<1 or w*h>int(config.get("max_source_pixels",40_000_000)):
                raise MemoryError("Full-frame native LibRaw pixel bound exceeded")
            rgb=raw.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16,
                                use_camera_wb=True,output_color=rawpy.ColorSpace.sRGB)
            if rgb.dtype!=np.uint16 or rgb.shape!=(h,w,3):
                raise ValueError("LibRaw returned unexpected shape or precision")
            black=list(raw.black_level_per_channel)
            white=int(raw.white_level)
        return {
            "linear_image_f32":rgb.astype(np.float32)/65535.0,
            "source_nef_sha256":source_sha,
            "color_space":"SceneLinear_sRGB","transfer":"LINEAR",
            "black_level":black,"white_level":white
        }
