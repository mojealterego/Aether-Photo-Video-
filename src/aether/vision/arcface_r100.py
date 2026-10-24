"""Explicit ArcFace R100 ONNX adapter with five-landmark alignment.

This module implements feature extraction/verification, NOT hallucination-proof
identity preservation. The caller supplies a real, licensed ArcFace R100 ONNX
file and five correctly ordered facial landmarks from a separate detector.
No bundled weights, auto-download, or hidden use of network services.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

ARC_FACE_TEMPLATE=np.array([
    [38.2946,51.6963], [73.5318,51.5014], [56.0252,71.7366],
    [41.5493,92.3655], [70.7299,92.2041]],dtype=np.float32)


def align_face_112(rgb:np.ndarray,landmarks:np.ndarray)->np.ndarray:
    a=np.asarray(rgb)
    pts=np.asarray(landmarks,dtype=np.float32)
    if a.ndim!=3 or a.shape[2]!=3 or a.dtype!=np.uint8 or min(a.shape[:2])<16:
        raise ValueError('Face image must be HWC uint8 RGB')
    if pts.shape!=(5,2) or not np.isfinite(pts).all():
        raise ValueError('Five finite facial landmarks required [left eye, right eye, nose, left mouth, right mouth]')
    if np.linalg.norm(pts[0]-pts[1])<2:
        raise ValueError('Eye landmarks are degenerate')
    trans, inliers=cv2.estimateAffinePartial2D(pts,ARC_FACE_TEMPLATE,method=cv2.LMEDS)
    if trans is None or not np.isfinite(trans).all() or abs(np.linalg.det(trans[:,:2]))<1e-6:
        raise ValueError('Could not align face landmarks')
    return cv2.warpAffine(a,trans,(112,112),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


class ArcFaceR100:
    """Extract unit L2 512-dimensional vectors using ONNX Runtime.

    Expects ONNX input [1,3,112,112], RGB normalized to [-1,+1]. Check
    model input ordering against the *specific* checkpoint's model card.
    OnnxRuntime is imported only on real inference; tests inject sessions.
    """
    def __init__(self,checkpoint:Path,*,providers:tuple[str,...]=('CPUExecutionProvider',),session=None):
        path=Path(checkpoint).expanduser().resolve()
        if path.suffix.lower()!='.onnx' or not path.is_file():
            raise FileNotFoundError('A local ArcFace R100 ONNX model file is required')
        if session is None:
            try:import onnxruntime as ort
            except ImportError as e:raise RuntimeError('Install onnxruntime on the execution host') from e
            available=set(ort.get_available_providers())
            if not set(providers).issubset(available):
                raise RuntimeError(f'ONNX provider unavailable: {set(providers)-available}')
            session=ort.InferenceSession(str(path),providers=list(providers))
        ins=session.get_inputs();outs=session.get_outputs()
        if len(ins)!=1 or len(outs)<1:
            raise ValueError('Unexpected ArcFace ONNX model input/output count')
        shape=ins[0].shape
        if len(shape)!=4 or any(isinstance(n,int) and n!=target
                                for n,target in zip(shape,(1,3,112,112))):
            raise ValueError('ArcFace R100 expected NCHW 1x3x112x112')
        self.input_name=ins[0].name
        self.output_name=outs[0].name
        self.session=session
        self.path=path

    def encode_aligned(self,rgb112:np.ndarray)->np.ndarray:
        image=np.asarray(rgb112)
        if image.shape!=(112,112,3) or image.dtype!=np.uint8:
            raise ValueError('ArcFace requires uint8 aligned RGB112')
        blob=((image.astype(np.float32)-127.5)/127.5).transpose(2,0,1)[None]
        output=self.session.run([self.output_name],{self.input_name:np.ascontiguousarray(blob)})
        vec=np.asarray(output[0],dtype=np.float32).reshape(-1)
        if vec.size!=512 or not np.isfinite(vec).all():
            raise ValueError('ArcFace output must be finite 512-d embedding')
        length=np.linalg.norm(vec)
        if length<1e-8:raise ValueError('Zero ArcFace embedding')
        return (vec/length).astype(np.float32)

    def encode(self,rgb:np.ndarray,landmarks:np.ndarray)->np.ndarray:
        return self.encode_aligned(align_face_112(rgb,landmarks))


def verify_arcface_pair(reference:np.ndarray,candidate:np.ndarray,*,threshold:float=.45)->tuple[bool,float]:
    """Similarity veto only; passing threshold does not establish real identity."""
    if not -1<=threshold<=1:raise ValueError('Similarity threshold outside [-1,1]')
    a=np.asarray(reference,dtype=np.float32).reshape(-1)
    b=np.asarray(candidate,dtype=np.float32).reshape(-1)
    if a.size!=512 or b.size!=512 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected two finite ArcFace 512-d embeddings')
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    if den<1e-8:raise ValueError('Cannot compare zero embeddings')
    sim=float(np.clip(np.dot(a,b)/den,-1,1))
    return sim>=threshold,sim