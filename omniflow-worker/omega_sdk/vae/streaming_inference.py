"""AETHERIS Ω-VAE disk-backed neural tiled inference for scene-linear RGB FP32.

Test-only unless a trusted runtime binds independently verified model
provenance and approval. Uses fixed halo and disjoint core publication.
A correct halo must cover the model's receptive field; this is a caller
contract, not inferred from the arbitrary torch module.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import os
import time
import numpy as np
import torch
from .tiling_plan import TilePlan,TilePlanner

class InferenceError(ValueError):pass

def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()

def run_tiled_neural(source:Path,destination:Path,model:torch.nn.Module,plan:TilePlan,
                     *,device:str='cpu',test_only:bool=False,approval=None)->dict:
    """Execute bounded one-tile-at-a-time FP32 RGB inference to .npy memmap.

    Production requires an external attested runtime, not a caller-supplied
    boolean/dict. This implementation intentionally refuses production.
    """
    if not test_only:
        raise InferenceError('PRODUCTION_DENIED: attested model admission and Control Plane E2E absent')
    source=Path(source);destination=Path(destination)
    if destination.exists() or destination.resolve()==source.resolve():
        raise FileExistsError('immutable input/output policy')
    if not isinstance(model,torch.nn.Module) or model.training:
        raise InferenceError('evaluated torch model required')
    if not isinstance(plan,TilePlan) or not TilePlanner.verify_coverage(plan):
        raise InferenceError('invalid tile plan')
    if not source.is_file() or source.suffix!='.npy' or destination.suffix!='.npy':
        raise InferenceError('disk-backed .npy arrays required')
    try:
        src=np.load(source,mmap_mode='r',allow_pickle=False)
    except (OSError,ValueError) as exc:
        raise InferenceError('invalid source array') from exc
    if src.ndim!=3 or src.shape[2]!=3 or src.dtype!=np.float32 or src.shape[:2]!=plan.image_size:
        raise InferenceError('finite scene-linear RGB FP32 HWC with exact plan dimensions required')
    if device!='cpu' and not (device.startswith('cuda') and torch.cuda.is_available()):
        raise InferenceError('requested accelerator unavailable')
    if next(model.parameters(),None) is not None:
        p=next(model.parameters())
        if p.device.type!=torch.device(device).type:raise InferenceError('model not on declared execution device')
    partial=destination.with_name(destination.name+'.partial.npy')
    if partial.exists():raise FileExistsError('uncommitted candidate already exists')
    start=time.perf_counter();tiles_done=0;out=None
    try:
        out=np.lib.format.open_memmap(partial,mode='w+',dtype='float32',shape=src.shape)
        with torch.inference_mode():
            for tile in plan.tiles:
                y0,y1,x0,x1=tile.padded_y0,tile.padded_y1,tile.padded_x0,tile.padded_x1
                raw=np.asarray(src[y0:y1,x0:x1,:],dtype=np.float32)
                if not np.isfinite(raw).all():raise InferenceError('nonfinite source tile')
                t=torch.from_numpy(np.ascontiguousarray(raw.transpose(2,0,1)))[None].to(device)
                prediction=model(t)
                if not isinstance(prediction,torch.Tensor) or prediction.shape!=t.shape or not torch.isfinite(prediction).all():
                    raise InferenceError('invalid neural output geometry or nonfinite prediction')
                rgb=prediction[0].permute(1,2,0).float().cpu().numpy()
                cy0,cy1=tile.core_y0,tile.core_y1
                cx0,cx1=tile.core_x0,tile.core_x1
                out[cy0:cy1,cx0:cx1,:]=rgb[cy0-y0:cy1-y0,cx0-x0:cx1-x0,:]
                tiles_done+=1
        out.flush();del out;out=None
        with partial.open('rb') as fd:os.fsync(fd.fileno())
        digest=_sha256(partial)
        os.replace(partial,destination)
    except Exception:
        if out is not None:del out
        partial.unlink(missing_ok=True)
        raise
    return {'status':'TEST_ONLY_UNAPPROVED','production_approved':False,
            'device':device,'image_shape':list(src.shape),
            'tiles_completed':tiles_done,'output_sha256':digest,
            'duration_seconds':round(time.perf_counter()-start,3),
            'gpu_e2e':False,'control_plane_e2e':False}
