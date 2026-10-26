"""Native OmniFlow Depth Anything 3 and CodeFormer nodes.

Inputs must be tenant-owned SHA-256 verified CAS artifacts. Model outputs
are UNVERIFIED hypotheses. Real weights are loaded from explicit local
checkpoints; commercial licensing and GPU evidence require separate approval.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import uuid
from typing import Any
import numpy as np
from PIL import Image
from .protocol import (
    ContentAddressedStorage, ExecutionContext, NodeExecutionMode,
    NodeManifest, OmegaNode, TrustClass, OMEGA_ABI_VERSION
)


def _verified_artifact(ctx:ExecutionContext,handle:dict[str,Any])->Path:
    if not isinstance(handle,dict) or handle.get("tenant_id")!=ctx.tenant_id:
        raise PermissionError("Invalid tenant-scoped artifact")
    expected=handle.get("sha256")
    if (not isinstance(expected,str) or len(expected)!=71
            or not expected.startswith("sha256:")):
        raise ValueError("Invalid artifact digest")
    root=ctx.cas_root.resolve()/"blobs"/ctx.tenant_id
    path=Path(handle["cas_path"])
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise PermissionError("Artifact not in scoped CAS")
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda:file.read(1<<20),b""):
            digest.update(part)
    if digest.hexdigest()!=expected[7:]:
        raise ValueError("Artifact integrity mismatch")
    return path


def _read_rgb8(path:Path,*,max_pixels:int=1_500_000)->np.ndarray:
    with Image.open(path) as image:
        if image.mode!="RGB" or image.width*image.height>max_pixels:
            raise ValueError("Explicit bounded display-sRGB RGB8 preview required")
        return np.array(image,dtype=np.uint8)


class DepthOmega(OmegaNode):
    """Raw relative depth and confidence; NEVER silently interpreted as meters."""
    def __init__(self,backend:Any,model_id:str="depth-anything/DA3-SMALL"):
        self.backend=backend
        self.model_id=model_id
    @classmethod
    def manifest(cls)->NodeManifest:
        return NodeManifest(
            "P006_DepthOmegaDA3","1.1.0","GEOMETRY",OMEGA_ABI_VERSION,
            NodeExecutionMode.PURE_DETERMINISTIC,TrustClass.T0_CORE_TRUSTED,
            ("depth-anything/DA3-SMALL",),False)
    def execute(self,ctx:ExecutionContext,inputs:dict[str,Any],
                config:dict[str,Any])->dict[str,Any]:
        file=_verified_artifact(ctx,inputs["image_artifact"])
        rgb=_read_rgb8(file,max_pixels=int(config.get("max_preview_pixels",1_500_000)))
        process_res=int(config.get("process_res",504))
        if not 224<=process_res<=1008:
            raise ValueError("Unbounded DA3 resolution")
        prediction=self.backend.inference([rgb],process_res=process_res)
        depth=np.asarray(prediction.depth,dtype=np.float32)
        confidence=np.asarray(prediction.conf,dtype=np.float32)
        if (depth.ndim!=3 or depth.shape[0]!=1
                or depth.shape[1:]!=confidence.shape[1:] or confidence.shape!=depth.shape):
            raise ValueError("DA3 prediction must return [1,H,W] depth and confidence")
        if (not np.isfinite(depth).all() or not np.isfinite(confidence).all()
                or min(depth.shape[1:])<1):
            raise ValueError("Nonfinite or empty DA3 output")
        staging=ctx.scratch_dir/(uuid.uuid4().hex+".npz")
        try:
            np.savez(staging,depth=depth[0],confidence_raw=confidence[0],
                     model_id=np.array(self.model_id),units=np.array("RELATIVE_NORMALIZED"))
            handle=ContentAddressedStorage(ctx.cas_root).commit_file_atomically(
                staging,ctx.tenant_id,"application/x-npz","RELATIVE_DEPTH")
        finally:
            staging.unlink(missing_ok=True)
        return dict(artifact_handle=asdict(handle),dimensions=list(depth.shape[1:]),
                    units="RELATIVE_NORMALIZED",confidence_encoding="RAW_MODEL_OUTPUT",
                    model_id=self.model_id,verification_status="UNVERIFIED")


class CodeFormerOmega(OmegaNode):
    """Candidate 512x512 face, pending independent biometric and operator gates."""
    def __init__(self,backend:Any):self.backend=backend
    @classmethod
    def manifest(cls)->NodeManifest:
        return NodeManifest(
            "F010_CodeFormerOmega","1.1.0","FACE_AND_IDENTITY",
            OMEGA_ABI_VERSION,NodeExecutionMode.PURE_DETERMINISTIC,
            TrustClass.T0_CORE_TRUSTED,("codeformer-v0.1.0",),False)
    def execute(self,ctx:ExecutionContext,inputs:dict[str,Any],
                config:dict[str,Any])->dict[str,Any]:
        weight=float(config.get("fidelity_weight",.5))
        if not math.isfinite(weight) or not 0<=weight<=1:
            raise ValueError("CodeFormer fidelity must be in [0,1]")
        path=_verified_artifact(ctx,inputs["image_artifact"])
        face=_read_rgb8(path,max_pixels=512*512)
        if face.shape!=(512,512,3):
            raise ValueError("Pre-aligned 512x512 RGB8 face crop required")
        restored=np.asarray(self.backend.restore_crop(face,w=weight))
        if restored.shape!=(512,512,3) or restored.dtype!=np.uint8:
            raise ValueError("CodeFormer returned invalid RGB8 crop")
        staging=ctx.scratch_dir/(uuid.uuid4().hex+".png")
        try:
            Image.fromarray(restored,"RGB").save(staging)
            handle=ContentAddressedStorage(ctx.cas_root).commit_file_atomically(
                staging,ctx.tenant_id,"image/png","display-sRGB")
        finally:
            staging.unlink(missing_ok=True)
        return dict(artifact_handle=asdict(handle),fidelity_weight=weight,
                    verification_status="NEEDS_REVIEW",identity_verified=False)


class LocalDA3Backend:
    """DA3 weights from a local snapshot, no automatic network fetch."""
    def __init__(self,snapshot_dir:Path,weight_sha256:str,device:str="cuda"):
        snapshot_dir=Path(snapshot_dir)
        weights=snapshot_dir/"model.safetensors"
        if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest()!=weight_sha256:
            raise ValueError("Missing or nonmatching local DA3 model.safetensors")
        import torch
        if device=="cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable")
        from depth_anything_3.api import DepthAnything3
        self.model=DepthAnything3.from_pretrained(str(snapshot_dir)).to(device=device).eval()
    def inference(self,images:list[np.ndarray],process_res:int=504):
        return self.model.inference(images,process_res=process_res,export_dir=None)


class LocalTorchCodeFormerBackend:
    """Original PyTorch CodeFormer checkpoint; license permission remains mandatory."""
    def __init__(self,checkpoint:Path,sha256:str,device:str="cuda"):
        checkpoint=Path(checkpoint)
        if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=sha256:
            raise ValueError("Missing or nonmatching codeformer.pth")
        import torch
        if device=="cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable")
        from basicsr.archs import codeformer_arch  # noqa: F401, register architecture
        from basicsr.utils.registry import ARCH_REGISTRY
        network=ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,codebook_size=1024,n_head=8,n_layers=9,
            connect_list=["32","64","128","256"]).to(device)
        checkpoint_data=torch.load(checkpoint,map_location="cpu",weights_only=True)
        network.load_state_dict(checkpoint_data["params_ema"],strict=True)
        self.network=network.eval()
        self.device=device
    def restore_crop(self,rgb:np.ndarray,*,w:float)->np.ndarray:
        import torch
        tensor=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()
        tensor=(tensor/127.5-1.0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            result=self.network(tensor,w=w,adain=True)[0]
        result=result[0].detach().to("cpu").float().clamp(-1,1)
        image=(result.permute(1,2,0).numpy()+1)*127.5
        return np.rint(image).clip(0,255).astype(np.uint8)
