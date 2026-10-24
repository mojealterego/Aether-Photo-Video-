"""Bounded-memory RGB16 ICC conversion with LittleCMS2.

Accepts only single-page contiguous, memmappable RGB8/RGB16 TIFF. Output
is a contiguous RGB16 sRGB BigTIFF, committed atomically after validation.
No RGB8 intermediate is used for 16-bit inputs.
"""
from __future__ import annotations
import ctypes as C
from ctypes.util import find_library
from pathlib import Path
import os
import numpy as np
from PIL import ImageCms
import tifffile

TYPE_RGB_16=(4<<16)|(3<<3)|2

class ICCUnavailable(RuntimeError):
    pass

class ICCTransform:
    def __init__(self,input_icc:bytes):
        if not input_icc or len(input_icc)>4*1024*1024:
            raise ValueError("Input ICC profile must be present and <=4 MiB")
        libname=find_library("lcms2") or os.getenv("AETHER_LCMS2_PATH")
        if not libname:
            raise ICCUnavailable("LittleCMS2 required; set AETHER_LCMS2_PATH if needed")
        self.lib=lib=C.CDLL(libname)
        lib.cmsOpenProfileFromMem.argtypes=[C.c_void_p,C.c_uint32]
        lib.cmsOpenProfileFromMem.restype=C.c_void_p
        lib.cmsCreate_sRGBProfile.argtypes=[]
        lib.cmsCreate_sRGBProfile.restype=C.c_void_p
        lib.cmsCreateTransform.argtypes=[C.c_void_p,C.c_uint32,C.c_void_p,C.c_uint32,C.c_uint32,C.c_uint32]
        lib.cmsCreateTransform.restype=C.c_void_p
        lib.cmsDoTransform.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint32]
        lib.cmsDeleteTransform.argtypes=[C.c_void_p]
        lib.cmsCloseProfile.argtypes=[C.c_void_p]
        self._profile_bytes=C.create_string_buffer(input_icc)
        self._src=lib.cmsOpenProfileFromMem(self._profile_bytes,len(input_icc))
        self._dst=None
        self._transform=None
        if not self._src: raise ValueError("Invalid source ICC profile")
        try:
            self._dst=lib.cmsCreate_sRGBProfile()
            if not self._dst: raise ValueError("Target sRGB profile unavailable")
            self._transform=lib.cmsCreateTransform(self._src,TYPE_RGB_16,self._dst,TYPE_RGB_16,0,0)
            if not self._transform: raise ValueError("Input ICC cannot convert RGB16 to sRGB")
        except BaseException:
            self.close()
            raise

    def convert(self,tile:np.ndarray)->np.ndarray:
        arr=np.asarray(tile)
        if (arr.ndim!=3 or arr.shape[2]!=3 or arr.dtype!=np.uint16 or
            not 1<=arr.shape[0]<=2048 or not 1<=arr.shape[1]<=2048):
            raise ValueError("Expected uint16 RGB tile <=2048 pixels per side")
        if self._transform is None: raise RuntimeError("ICC transform closed")
        src=np.ascontiguousarray(arr)
        result=np.empty_like(src)
        self.lib.cmsDoTransform(self._transform,src.ctypes.data,result.ctypes.data,src.shape[0]*src.shape[1])
        return result

    def close(self)->None:
        if getattr(self,"_transform",None):
            self.lib.cmsDeleteTransform(self._transform);self._transform=None
        if getattr(self,"_dst",None):
            self.lib.cmsCloseProfile(self._dst);self._dst=None
        if getattr(self,"_src",None):
            self.lib.cmsCloseProfile(self._src);self._src=None
    def __enter__(self): return self
    def __exit__(self,*args): self.close()

def srgb_profile()->bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

def rgb16_tiff_to_srgb(source:str|Path,output:str|Path,*,rows:int=64,
                       assume_srgb:bool=False,max_pixels:int=225_000_000)->dict:
    """Transform RGB TIFF in bounded strips; no RAW demosaicing is implied."""
    source,output=Path(source),Path(output)
    if not source.is_file() or source.resolve()==output.resolve() or output.exists():
        raise ValueError("Missing input or unsafe output path")
    if source.suffix.lower() not in (".tif",".tiff") or output.suffix.lower() not in (".tif",".tiff"):
        raise ValueError("Input and output must be TIFF")
    if not 1<=rows<=256 or not 1<=max_pixels<=1_000_000_000:
        raise ValueError("Invalid strip or pixel cap")
    with tifffile.TiffFile(source) as tif:
        if len(tif.pages)!=1: raise ValueError("Single-page TIFF required")
        p=tif.pages[0]
        if p.dtype not in (np.dtype("uint8"),np.dtype("uint16")) or len(p.shape)!=3 or p.shape[2]!=3:
            raise ValueError("RGB uint8/uint16 TIFF required")
        if p.photometric!=tifffile.PHOTOMETRIC.RGB:
            raise ValueError("Input must be RGB, not CFA RAW")
        orientation=p.tags.get("Orientation")
        if orientation is not None and orientation.value!=1:
            raise ValueError("Normalize TIFF orientation first")
        h,w=p.shape[:2]
        if h*w>max_pixels or rows*w*3*2>128*1024*1024:
            raise ValueError("Input/stripe exceeds mobile memory budget")
        tag=p.tags.get("InterColorProfile") or p.tags.get("ICCProfile")
        icc=bytes(tag.value) if tag is not None else None
    if not icc and not assume_srgb:
        raise ValueError("No source ICC profile; specify assume_srgb explicitly")
    try: mapped=tifffile.memmap(source)
    except (ValueError,OSError) as ex:
        raise ValueError("TIFF must be contiguous, uncompressed and memmappable") from ex
    if mapped.shape!=(h,w,3): raise ValueError("Unexpected TIFF memory map")
    output.parent.mkdir(parents=True,exist_ok=True)
    part=output.with_name(output.name+".partial.tif")
    if part.exists(): raise FileExistsError("Interrupted partial output exists")
    dest=None
    try:
        dest=tifffile.memmap(part,shape=(h,w,3),dtype=np.uint16,bigtiff=True,
                             photometric="rgb",metadata=None,iccprofile=srgb_profile())
        transform=ICCTransform(icc) if icc is not None else None
        try:
            for y in range(0,h,rows):
                y1=min(y+rows,h)
                patch=np.asarray(mapped[y:y1],dtype=np.uint16)
                if mapped.dtype==np.uint8: patch=patch*np.uint16(257)
                if transform: patch=transform.convert(patch)
                dest[y:y1]=patch
            dest.flush()
        finally:
            if transform: transform.close()
        del dest;dest=None
        with tifffile.TiffFile(part) as check:
            page=check.pages[0]
            if not check.is_bigtiff or page.shape!=(h,w,3) or page.dtype!=np.dtype("uint16"):
                raise RuntimeError("Final RGB16 validation failed")
        os.replace(part,output)
    finally:
        if dest is not None: dest.flush()
    return dict(input=str(source),output=str(output),size=[w,h],
                input_icc_present=bool(icc),destination="RGB16 sRGB BigTIFF",
                conversion="LittleCMS2 RGB16" if icc else "explicit_assume_srgb")
