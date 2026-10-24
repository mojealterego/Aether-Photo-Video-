"""Bounded RAW/DNG Bayer CFA16 decoder. Strict, explicit supported subset.

Supports only one-page, uncompressed/memmappable uint16 Bayer (RGGB/GRBG/
GBRG/BGGR), with BlackLevel, WhiteLevel, AsShotNeutral and ColorMatrix1.
Rejects compression, orientation, active crop and DNG OpcodeLists rather than
claiming incorrect RAW fidelity. Writes contiguous sRGB RGB16 BigTIFF.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import tifffile
from PIL import ImageCms

XYZ_TO_SRGB=np.array([[3.2406,-1.5372,-.4986],[-.9689,1.8758,.0415],[.0557,-.2040,1.0570]],dtype='f8')

class RAWUnsupportedError(ValueError):
    pass

def _tag(page,name):
    t=page.tags.get(name)
    return t.value if t is not None else None

def _camera_settings(page):
    if page.photometric!=tifffile.PHOTOMETRIC.CFA or len(page.shape)!=2:
        raise RAWUnsupportedError('Require one-plane Bayer DNG')
    dim=_tag(page,'CFARepeatPatternDim')
    cf=_tag(page,'CFAPattern')
    if dim is None or tuple(dim)!=(2,2) or cf is None:
        raise RAWUnsupportedError('Require 2x2 Bayer')
    pattern=np.asarray(list(cf),dtype='u1').ravel()
    if len(pattern)!=4 or sorted(pattern.tolist())!=[0,1,1,2]:
        raise RAWUnsupportedError('Unsupported CFA')
    items=[_tag(page,x) for x in ('BlackLevel','WhiteLevel','AsShotNeutral','ColorMatrix1')]
    if any(v is None for v in items):
        raise RAWUnsupportedError('Missing DNG black/white/WB/ColorMatrix1')
    black,white,neutral,matrix=map(lambda v:np.asarray(v,dtype='f8').ravel(),items)
    if black.size==1:black=np.repeat(black,4)
    if white.size==1:white=np.repeat(white,4)
    if (black.size!=4 or white.size!=4 or neutral.size!=3 or matrix.size!=9
        or not all(np.isfinite(x).all() for x in (black,white,neutral,matrix))
        or np.any(white<=black) or np.any(neutral<=0)):
        raise RAWUnsupportedError('Invalid DNG calibration values')
    matrix=matrix.reshape(3,3)
    if abs(float(np.linalg.det(matrix)))<1e-6:
        raise RAWUnsupportedError('Singular color matrix')
    return pattern.reshape(2,2),black.astype('f4'),white.astype('f4'),(neutral[1]/neutral).astype('f4'),(XYZ_TO_SRGB@np.linalg.inv(matrix)).astype('f4')

def _bayer_linear(sensor,x0,y0,pattern,black,white):
    h,w=sensor.shape
    yy=(np.arange(y0,y0+h)%2).astype('i4')
    xx=(np.arange(x0,x0+w)%2).astype('i4')
    phase=2*yy[:,None]+xx[None,:]
    colors=pattern[yy[:,None],xx[None,:]]
    lum=np.clip((sensor.astype('f4')-black[phase])/(white[phase]-black[phase]),0,1)
    kernel=np.array([[1,2,1],[2,4,2],[1,2,1]],'f4')
    output=np.empty((h,w,3),'f4')
    for channel in range(3):
        known=(colors==channel).astype('f4')
        src=np.pad(lum*known,1,mode='edge')
        mask=np.pad(known,1,mode='edge')
        num=np.zeros((h,w),'f4')
        den=np.zeros((h,w),'f4')
        for j in range(3):
            for i in range(3):
                num+=kernel[j,i]*src[j:j+h,i:i+w]
                den+=kernel[j,i]*mask[j:j+h,i:i+w]
        output[:,:,channel]=np.where(known!=0,lum,num/np.maximum(den,1e-6))
    return output

def _srgb(linear):
    x=np.clip(linear,0,1)
    return np.where(x<=.0031308,12.92*x,1.055*np.power(x,1/2.4)-.055)

def decode_uncompressed_dng(source:str|Path,output:str|Path,*,tile_size:int=256,
                            max_pixels:int=250_000_000,correct_white_balance:bool=True)->dict:
    source,output=Path(source),Path(output)
    if source.suffix.lower()!='.dng' or output.suffix.lower() not in ('.tif','.tiff'):
        raise RAWUnsupportedError('Input .dng and output .tif required')
    if not source.is_file() or output.exists() or source.resolve()==output.resolve():
        raise ValueError('Missing source, existing result or source/output collision')
    if type(tile_size)!=int or not 16<=tile_size<=1024 or max_pixels<1:
        raise ValueError('Invalid tile or image budget')
    with tifffile.TiffFile(source) as tif:
        if len(tif.pages)!=1:raise RAWUnsupportedError('Multi-page DNG needs LibRaw')
        page=tif.pages[0]
        if page.dtype!=np.dtype('uint16'):
            raise RAWUnsupportedError('Only unpacked uint16 CFA is supported')
        if _tag(page,'Orientation') not in (None,1):
            raise RAWUnsupportedError('DNG orientation unsupported')
        if any(_tag(page,t) is not None for t in ('OpcodeList1','OpcodeList2','OpcodeList3','LinearizationTable','ActiveArea')):
            raise RAWUnsupportedError('Camera DNG opcodes/active area unsupported')
        pattern,black,white,wb,matrix=_camera_settings(page)
        h,w=page.shape
        if h*w>max_pixels:raise MemoryError('DNG image pixel limit exceeded')
    try:sensor=tifffile.memmap(source)
    except (ValueError,OSError) as exc:
        raise RAWUnsupportedError('Compressed/packed/tiled DNG needs LibRaw') from exc
    if sensor.shape!=(h,w) or sensor.dtype!=np.dtype('uint16'):
        raise RAWUnsupportedError('Sensor memory layout mismatch')
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(output.name+'.partial.tif')
    if partial.exists():raise FileExistsError('Unresolved DNG partial output')
    dst=None
    try:
        dst=tifffile.memmap(partial,shape=(h,w,3),dtype='u2',bigtiff=True,
                           photometric='rgb',metadata=None,iccprofile=icc)
        for y0 in range(0,h,tile_size):
            y1=min(h,y0+tile_size)
            for x0 in range(0,w,tile_size):
                x1=min(w,x0+tile_size)
                sx0,sy0=max(0,x0-2),max(0,y0-2)
                sx1,sy1=min(w,x1+2),min(h,y1+2)
                patch=_bayer_linear(sensor[sy0:sy1,sx0:sx1],sx0,sy0,pattern,black,white)
                patch=patch[y0-sy0:y1-sy0,x0-sx0:x1-sx0]
                if correct_white_balance:patch*=wb[None,None,:]
                linear=np.einsum('...c,kc->...k',patch,matrix,optimize=True)
                dst[y0:y1,x0:x1]=np.rint(_srgb(linear)*65535).astype('u2')
            dst.flush()
        del dst;dst=None
        with tifffile.TiffFile(partial) as check:
            p=check.pages[0]
            if not check.is_bigtiff or p.shape!=(h,w,3) or p.dtype!=np.dtype('uint16'):
                raise RuntimeError('Final DNG conversion failed validation')
        os.replace(partial,output)
    finally:
        if dst is not None:dst.flush()
    return {'input':str(source),'output':str(output),'dimensions':[w,h],
            'demosaicing':'normalized_bilinear_bayer',
            'profile':'ColorMatrix1/AsShotNeutral -> sRGB',
            'full_dng_coverage':False}
