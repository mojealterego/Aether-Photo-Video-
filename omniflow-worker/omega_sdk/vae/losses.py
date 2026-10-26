"""Differentiable training losses for Aetheris Ω-VAE (partial CompositeLossOmega).

Photographic mode defaults to ZERO adversarial loss. No learned perceptual
metric, optical-flow warp or CIEDE2000 is silently substituted.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import ContractError
from .neural import DiagonalGaussian

@dataclass(frozen=True)
class LossWeights:
    linear:float=1.0
    log_luma:float=.5
    multiscale:float=.25
    focal_frequency:float=.15
    scharr_edge:float=.2
    beta_kl:float=1e-6
    kl_free_bits:float=.01
    uncertainty_nll:float=.05
    seam_consistency:float=.5
    adversarial:float=0.0
    luminance_y0:float=.18
    sensor_noise_floor:float=1e-4

    def validate(self):
        for name,value in vars(self).items():
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ContractError(f'invalid loss coefficient: {name}')
        if self.adversarial!=0:
            raise ContractError('P16 documentary profile forbids adversarial training weight')
        if self.luminance_y0<=0:
            raise ContractError('positive luminance scale required')


class CompositeLossOmega(nn.Module):
    """Subset of Argon's composite target; unsupported terms are explicit absent."""
    def __init__(self,weights:LossWeights|None=None):
        super().__init__()
        self.weights=weights or LossWeights()
        self.weights.validate()
        k=torch.tensor([[-3,0,3],[-10,0,10],[-3,0,3]],dtype=torch.float32)/32
        self.register_buffer('scharr_x',k)
        self.register_buffer('scharr_y',k.T.contiguous())
        self.register_buffer('aces_luma',torch.tensor([.27222872,.67408177,.05368952],dtype=torch.float32))

    def _same(self,a,b):
        if (a.ndim!=4 or a.shape!=b.shape or a.shape[1]!=3 or
            not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise ContractError('finite matching NCHW scene-linear RGB required')
        return a.float(),b.float()

    def charbonnier(self,a,b):
        x,y=self._same(a,b)
        diff=(x-y).abs().sub(self.weights.sensor_noise_floor).clamp_min(0)
        return (diff.square()+1e-6).sqrt().sub(1e-3).mean()

    def log_luma(self,a,b):
        x,y=self._same(a,b)
        weights=self.aces_luma.to(x.device).view(1,3,1,1)
        left=(x*weights).sum(dim=1,keepdim=True)
        right=(y*weights).sum(dim=1,keepdim=True)
        def signed(v):return v.sign()*torch.log1p(v.abs()/self.weights.luminance_y0)
        return (signed(left)-signed(right)).abs().mean()

    def multiscale(self,a,b,levels=3):
        x,y=self._same(a,b)
        total=x.new_zeros(())
        iterations=0
        for _ in range(levels):
            if min(x.shape[-2:])<4:break
            dx=F.avg_pool2d(x,2);dy=F.avg_pool2d(y,2)
            xr=x-F.interpolate(dx,size=x.shape[-2:],mode='bilinear',align_corners=False)
            yr=y-F.interpolate(dy,size=y.shape[-2:],mode='bilinear',align_corners=False)
            total=total+F.l1_loss(xr,yr)
            iterations+=1;x,y=dx,dy
        return (total+F.l1_loss(x,y))/(iterations+1)

    def frequency(self,a,b):
        x,y=self._same(a,b)
        h,w=x.shape[-2:]
        if min(h,w)<2:raise ContractError('FFT window too small')
        wy=torch.hann_window(h,periodic=False,device=x.device,dtype=torch.float32)
        wx=torch.hann_window(w,periodic=False,device=x.device,dtype=torch.float32)
        win=torch.outer(wy,wx)[None,None]
        spectrum=torch.fft.rfft2(x*win,norm='ortho')-torch.fft.rfft2(y*win,norm='ortho')
        sq=spectrum.real.square()+spectrum.imag.square()
        focal=(sq+1e-8).sqrt().detach()
        focal=focal/(focal.amax(dim=(-2,-1),keepdim=True)+1e-8)
        return (focal*sq).mean()

    def edges(self,a,b):
        x,y=self._same(a,b)
        def grad(z):
            log=z.sign()*torch.log1p(z.abs()/self.weights.luminance_y0)
            ch=log.shape[1]
            zpad=F.pad(log,(1,1,1,1),mode='replicate')
            kx=self.scharr_x.to(z.device).view(1,1,3,3).repeat(ch,1,1,1)
            ky=self.scharr_y.to(z.device).view(1,1,3,3).repeat(ch,1,1,1)
            return F.conv2d(zpad,kx,groups=ch),F.conv2d(zpad,ky,groups=ch)
        gx,gy=grad(x);hx,hy=grad(y)
        return F.l1_loss(gx,hx)+F.l1_loss(gy,hy)

    def forward(self,prediction,ground_truth,posterior:DiagonalGaussian,
                *,variance:torch.Tensor|None=None,
                overlap:tuple[torch.Tensor,torch.Tensor]|None=None):
        if not isinstance(posterior,DiagonalGaussian):
            raise ContractError('posterior object is required')
        x,y=self._same(prediction,ground_truth)
        terms={
            'linear':self.charbonnier(x,y),
            'log_luma':self.log_luma(x,y),
            'multiscale':self.multiscale(x,y),
            'focal_frequency':self.frequency(x,y),
            'scharr_edge':self.edges(x,y),
            'kl':posterior.kl(self.weights.kl_free_bits),
        }
        weighted=(terms['linear']*self.weights.linear+
                  terms['log_luma']*self.weights.log_luma+
                  terms['multiscale']*self.weights.multiscale+
                  terms['focal_frequency']*self.weights.focal_frequency+
                  terms['scharr_edge']*self.weights.scharr_edge+
                  terms['kl']*self.weights.beta_kl)
        if variance is not None:
            if variance.shape!=x.shape or not torch.isfinite(variance).all() or torch.any(variance<=0):
                raise ContractError('finite positive variance tensor required')
            s=variance.float().clamp(1e-6,1e3)
            terms['uncertainty_nll']=(.5*((x-y).square()/s+s.log())).mean()
            weighted=weighted+self.weights.uncertainty_nll*terms['uncertainty_nll']
        if overlap is not None:
            if len(overlap)!=2 or overlap[0].shape!=overlap[1].shape or not all(torch.isfinite(z).all() for z in overlap):
                raise ContractError('invalid tile overlap tensors')
            terms['seam_consistency']=F.l1_loss(overlap[0].float(),overlap[1].float())
            weighted=weighted+self.weights.seam_consistency*terms['seam_consistency']
        if not torch.isfinite(weighted):
            raise ContractError('nonfinite composite training loss')
        return weighted,{key:float(value.detach().item()) for key,value in terms.items()}
