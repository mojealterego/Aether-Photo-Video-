"""Optional bounded-memory LibRaw/rawpy decoder for complex camera RAW/DNG.

LibRaw's full-frame postprocess IS NOT out-of-core; physical S24 RAM
profiling is mandatory before raising the conservative memory cap.
"""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import tifffile
from PIL import ImageCms

class UnsupportedRawMemory(MemoryError):pass

def decode_libraw_safe(source:str|Path,output:str|Path,*,max_source_pixels:int=40_000_000,
                       max_decoded_mb:int=900,reserve_free_gb:float=1.0)->dict:
    source,output=Path(source),Path(output)
    if not source.is_file() or output.exists() or source.resolve()==output.resolve():
        raise ValueError('Missing input, unsafe overwrite or existing result')
    if not 0<max_source_pixels<=250_000_000 or max_decoded_mb<100 or reserve_free_gb<0:
        raise ValueError('Unsafe LibRaw memory or disk limits')
    try:import rawpy
    except ImportError as err:
        raise RuntimeError('Install LibRaw/rawpy compatible with the execution platform') from err
    output.parent.mkdir(parents=True,exist_ok=True)
    with rawpy.imread(str(source)) as decoder:
        w,h=int(decoder.sizes.width),int(decoder.sizes.height)
        if min(w,h)<1 or w*h>max_source_pixels:
            raise UnsupportedRawMemory('Native full-frame RAW exceeds pixel limit')
        if (w*h*24)/(1024**2)>max_decoded_mb:
            raise UnsupportedRawMemory('Predicted LibRaw memory exceeds cap')
        vfs=os.statvfs(output.parent)
        if vfs.f_bavail*vfs.f_frsize<w*h*6+int(reserve_free_gb*1024**3):
            raise OSError('Insufficient output disk reserve')
        rgb=decoder.postprocess(output_bps=16,output_color=rawpy.ColorSpace.sRGB,
                                use_camera_wb=True,no_auto_bright=True,gamma=(2.222,4.5))
        if rgb.shape!=(h,w,3) or rgb.dtype!=np.dtype('uint16'):
            raise ValueError('Refuse incorrect output or implicit RGB8 quantization')
        icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        partial=output.with_name(output.name+'.partial.tif')
        if partial.exists():raise FileExistsError('Previous incomplete RAW output')
        mm=None
        try:
            mm=tifffile.memmap(partial,shape=(h,w,3),dtype='u2',bigtiff=True,
                              photometric='rgb',metadata=None,iccprofile=icc)
            for y in range(0,h,64):mm[y:y+64]=rgb[y:y+64]
            mm.flush();del mm;mm=None
            with tifffile.TiffFile(partial) as tif:
                if (not tif.is_bigtiff or tif.pages[0].dtype!=np.dtype('uint16')
                    or tif.pages[0].shape!=(h,w,3)):
                    raise RuntimeError('Invalid LibRaw RGB16 BigTIFF')
            os.replace(partial,output)
        finally:
            if mm is not None:mm.flush()
    return {'source':str(source),'output':str(output),'dimensions':[w,h],
            'source_color':'LibRaw camera to sRGB','bits_per_channel':16,
            'full_frame_native_allocation':True,'oom_free_guaranteed':False}
