"""Reproducible relative ΔE00 QC for an ICC-tagged RGB16 TIFF and preview.

This is NOT absolute camera calibration. Without a color target and known
illuminant, ΔE00 here quantifies two renders, not camera color accuracy.
"""
from __future__ import annotations
import hashlib
from io import BytesIO
from pathlib import Path
import numpy as np
import cv2
import tifffile
from PIL import Image,ImageCms
from .deltae import ciede2000

class ColorEvidenceError(ValueError):pass

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def measure_preview_deltae(reference_tiff:Path,candidate_jpeg:Path,*,assume_candidate_srgb:bool=False)->dict:
    reference_tiff=Path(reference_tiff);candidate_jpeg=Path(candidate_jpeg)
    with tifffile.TiffFile(reference_tiff) as tf:
        page=tf.pages[0]
        if page.dtype!=np.uint16 or len(page.shape)!=3 or page.shape[2]!=3:
            raise ColorEvidenceError('RGB16 TIFF reference required')
        tag=page.tags.get('InterColorProfile')
        if tag is None:raise ColorEvidenceError('reference ICC profile missing')
        try:
            profile=ImageCms.ImageCmsProfile(BytesIO(tag.value))
            icc_name=ImageCms.getProfileName(profile).strip()
        except Exception as exc:raise ColorEvidenceError('invalid reference ICC') from exc
        if 'srgb' not in icc_name.lower():
            raise ColorEvidenceError('this measurement supports sRGB ICC references only')
    try:
        source=tifffile.memmap(reference_tiff)
    except Exception as exc:raise ColorEvidenceError('reference must be memmappable') from exc
    with Image.open(candidate_jpeg) as im:
        if im.format!='JPEG' or im.mode!='RGB' or im.width*im.height>20_000_000:
            raise ColorEvidenceError('bounded RGB JPEG required')
        if im.info.get('icc_profile'):
            cp=ImageCms.ImageCmsProfile(BytesIO(im.info['icc_profile']))
            if 'srgb' not in ImageCms.getProfileName(cp).lower():
                raise ColorEvidenceError('unsupported candidate ICC profile')
        elif not assume_candidate_srgb:
            raise ColorEvidenceError('untagged JPEG requires explicit sRGB assumption')
        candidate=np.asarray(im,dtype=np.float32)/255.
    ref=cv2.resize(source,(candidate.shape[1],candidate.shape[0]),interpolation=cv2.INTER_AREA).astype(np.float32)/65535.
    labref=cv2.cvtColor(ref,cv2.COLOR_RGB2LAB)
    labcandidate=cv2.cvtColor(candidate,cv2.COLOR_RGB2LAB)
    diff=ciede2000(labref,labcandidate)
    if not np.isfinite(diff).all():raise ColorEvidenceError('invalid color difference result')
    return {'reference_file':reference_tiff.name,'reference_sha256':_sha(reference_tiff),
            'candidate_file':candidate_jpeg.name,'candidate_sha256':_sha(candidate_jpeg),
            'reference_icc_name':icc_name,'candidate_color_assumption':'sRGB D65',
            'pixels_compared':int(diff.size),'mean_deltaE00':float(np.mean(diff)),
            'median_deltaE00':float(np.median(diff)),
            'p95_deltaE00':float(np.percentile(diff,95)),
            'p99_deltaE00':float(np.percentile(diff,99)),
            'max_deltaE00':float(np.max(diff)),
            'absolute_color_accuracy_certified':False,
            'method':'OpenCV sRGB to CIE Lab D65; CIEDE2000, TIFF downsample INTER_AREA'}
