"""Optional ONNX relative-depth adapter for bounded Android RGB previews.

Requires a separately supplied compatible Depth Anything ONNX export and
licensed model weights. Produces RELATIVE, not physically calibrated depth.
No native Qualcomm QNN acceleration is implied.
"""
from __future__ import annotations
import numpy as np
from PIL import Image

class DepthAnyOnnxAdapter:
    def __init__(self, session, *, near_is_higher:bool=True, max_preview_pixels:int=1_000_000):
        self.session=session
        self.near_is_higher=near_is_higher
        self.max_preview_pixels=max_preview_pixels
        first=session.get_inputs()[0]
        self.input_name=first.name
        shape=first.shape
        if len(shape)!=4 or shape[1]!=3:
            raise ValueError('ONNX model must accept NCHW RGB')
        h,w=shape[2:4]
        self.size=(w,h) if type(h) is int and type(w) is int else (518,518)
        if min(self.size)<32 or max(self.size)>1024:
            raise ValueError('Unsupported model input shape')

    @staticmethod
    def _resize(a:np.ndarray,width:int,height:int)->np.ndarray:
        return np.stack([
            np.asarray(Image.fromarray(a[:,:,c],mode='F').resize(
                (width,height),Image.Resampling.BILINEAR),dtype='f4')
            for c in range(a.shape[2])],axis=-1)

    def predict(self,preview_rgb:np.ndarray)->np.ndarray:
        a=np.asarray(preview_rgb,dtype=np.float32)
        if (a.ndim!=3 or a.shape[-1]!=3 or min(a.shape[:2])<2 or
            a.shape[0]*a.shape[1]>self.max_preview_pixels or
            not np.isfinite(a).all() or a.min()<0 or a.max()>1):
            raise ValueError('Provide bounded finite RGB preview in [0,1]')
        h,w=a.shape[:2]
        tw,th=self.size
        scale=min(tw/w,th/h)
        ww=max(1,min(tw,round(w*scale)))
        hh=max(1,min(th,round(h*scale)))
        scaled=self._resize(a,ww,hh)
        x0,y0=(tw-ww)//2,(th-hh)//2
        padded=np.zeros((th,tw,3),np.float32)
        padded[y0:y0+hh,x0:x0+ww]=scaled
        means=np.array([.485,.456,.406],np.float32)
        stds=np.array([.229,.224,.225],np.float32)
        tensor=np.transpose((padded-means)/stds,(2,0,1))[None].astype('f4')
        output=np.asarray(self.session.run(None,{self.input_name:tensor})[0],dtype='f4')
        while output.ndim>2 and output.shape[0]==1:
            output=output[0]
        if output.ndim!=2 or not np.isfinite(output).all() or min(output.shape)<2:
            raise ValueError('Unsupported ONNX depth output')
        oh,ow=output.shape
        xx0=round(x0*ow/tw);xx1=round((x0+ww)*ow/tw)
        yy0=round(y0*oh/th);yy1=round((y0+hh)*oh/th)
        cropped=output[yy0:yy1,xx0:xx1]
        if min(cropped.shape)<2:
            raise ValueError('Degenerate unpadded depth output')
        mn,mx=float(cropped.min()),float(cropped.max())
        normalized=np.zeros_like(cropped) if mx-mn<1e-8 else (cropped-mn)/(mx-mn)
        if not self.near_is_higher:
            normalized=1-normalized
        return np.asarray(Image.fromarray(normalized,mode='F').resize(
            (w,h),Image.Resampling.BILINEAR),dtype='f4').clip(0,1)


def relative_depth_focus_mask(preview_rgb:np.ndarray, depth:np.ndarray, *,
                              focus_depth:float=.5, tolerance:float=.15,
                              texture_weight:float=.35)->np.ndarray:
    """Bounded relative-depth gate for a chosen focus plane; no metric DoF claim.

    Output is a normalized 2D MASK. Can be used by the mobile renderer or
    optionally refined with the core's guided filter.
    """
    preview=np.asarray(preview_rgb,dtype=np.float32)
    d=np.asarray(depth,dtype=np.float32)
    if (preview.ndim!=3 or preview.shape[2]!=3 or d.shape!=preview.shape[:2] or
        not np.isfinite(preview).all() or not np.isfinite(d).all() or
        preview.min()<0 or preview.max()>1 or d.min()<0 or d.max()>1):
        raise ValueError('Expected registered normalized preview and depth')
    if not (0<=focus_depth<=1 and .01<=tolerance<=1 and 0<=texture_weight<=1):
        raise ValueError('Invalid focus plane or texture weight')
    gate=np.exp(-.5*((d-focus_depth)/tolerance)**2).astype(np.float32)
    if texture_weight:
        luma=.2126*preview[:,:,0]+.7152*preview[:,:,1]+.0722*preview[:,:,2]
        gy,gx=np.gradient(luma)
        magnitude=np.sqrt(gx*gx+gy*gy)
        threshold=max(float(np.percentile(magnitude,98)),1e-6)
        texture=np.clip(magnitude/threshold,0,1)
        gate*=1-texture_weight+texture_weight*texture
    return np.clip(gate,0,1).astype(np.float32)
