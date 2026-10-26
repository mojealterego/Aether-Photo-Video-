"""AETHERIS Ω-VAE 2D building blocks, untrained weights; NOT production inference.

SYS-AETHERIS-VISION-VAE-017: f_s=8, scene-linear output, Haar path,
window-local attention, global color context and causality primitives.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import ContractError


class BlurPool2D(nn.Module):
    def __init__(self,channels:int):
        super().__init__()
        if channels<1:raise ValueError('channels must be positive')
        one=torch.tensor([1.,2.,1.])
        filt=torch.outer(one,one);filt/=filt.sum()
        self.register_buffer('weight',filt.view(1,1,3,3).repeat(channels,1,1,1))
        self.channels=channels

    def forward(self,x):
        return F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),self.weight.to(x.dtype),groups=self.channels)


class HaarMicroPath(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        basis=torch.tensor([[[.5,.5],[.5,.5]],[[-.5,-.5],[.5,.5]],[[-.5,.5],[-.5,.5]],[[.5,-.5],[-.5,.5]]])
        self.register_buffer('basis',basis.unsqueeze(1).repeat(in_channels,1,1,1))
        self.project=nn.Conv2d(in_channels*4,out_channels,3,padding=1)
        self.in_channels=in_channels

    def forward(self,x):
        if x.shape[-1]%2 or x.shape[-2]%2:
            x=F.pad(x,(0,x.shape[-1]%2,0,x.shape[-2]%2),mode='replicate')
        return self.project(F.conv2d(x,self.basis.to(x.dtype),stride=2,groups=self.in_channels))


class LocalAttention2D(nn.Module):
    def __init__(self,channels:int,window:int=8,heads:int=4):
        super().__init__()
        if channels%heads or window<2 or window>32:raise ValueError('invalid windowed attention')
        self.window=window;self.heads=heads;self.dim=channels//heads
        self.norm=nn.GroupNorm(math.gcd(channels,8),channels)
        self.qkv=nn.Conv2d(channels,channels*3,1)
        self.proj=nn.Conv2d(channels,channels,1)

    def forward(self,x):
        batch,channels,h,w=x.shape;win=self.window
        ph=(-h)%win;pw=(-w)%win
        p=F.pad(x,(0,pw,0,ph),mode='replicate')
        hp,wp=p.shape[-2:];nh=hp//win;nw=wp//win
        qkv=self.qkv(self.norm(p)).reshape(batch,3,self.heads,self.dim,nh,win,nw,win)
        qkv=qkv.permute(1,0,4,6,2,5,7,3).reshape(3,batch*nh*nw,self.heads,win*win,self.dim)
        q,k,v=qkv.unbind(0)
        result=F.scaled_dot_product_attention(q,k,v)
        result=result.reshape(batch,nh,nw,self.heads,win,win,self.dim)
        result=result.permute(0,3,6,1,4,2,5).reshape(batch,channels,hp,wp)
        return (p+self.proj(result))[:,:,:h,:w]


class DiagonalGaussian:
    def __init__(self,moments:torch.Tensor):
        if moments.ndim!=4 or moments.shape[1]%2 or not torch.isfinite(moments).all():
            raise ContractError('invalid VAE posterior moments')
        mu,logvar=moments.float().chunk(2,dim=1)
        self.mean=mu;self.logvar=logvar.clamp(-30,20)

    def mode(self):return self.mean
    def sample(self,seed:int|None=None):
        gen=torch.Generator(device=self.mean.device)
        if seed is not None:gen.manual_seed(int(seed))
        noise=torch.randn(self.mean.shape,device=self.mean.device,dtype=self.mean.dtype,
                          generator=gen if seed is not None else None)
        return self.mean+noise*torch.exp(.5*self.logvar)
    def kl(self,free_bits:float=0.):
        if not math.isfinite(free_bits) or free_bits<0:raise ValueError('invalid KL free bits')
        term=.5*(self.mean.square()+self.logvar.exp()-1-self.logvar)
        per_channel=term.mean(dim=(0,2,3)).clamp_min(0)
        return per_channel.clamp_min(free_bits).sum()


class GlobalColorSummary(nn.Module):
    def __init__(self,channels:int=3,context_dim:int=32):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(channels*4,context_dim),nn.SiLU(),nn.Linear(context_dim,context_dim))
    def forward(self,x):
        p=F.adaptive_avg_pool2d(x.float(),(8,8))
        features=torch.cat((p.mean((-2,-1)),p.std((-2,-1),unbiased=False),
                            p.amin((-2,-1)),p.amax((-2,-1))),dim=1)
        return self.net(features)


class VAE2D(nn.Module):
    """A small parametrizable untrained 2D baseline, P16/G16/E8 topology.

    Not certified for any profile; released weights and training data absent.
    """
    def __init__(self,channels:int=16,base:int=32,context_dim:int=32):
        super().__init__()
        if channels<1 or base<8 or base%4:raise ValueError('invalid VAE geometry')
        self.context=GlobalColorSummary(3,context_dim)
        self.stem=nn.Conv2d(3,base,3,padding=1)
        self.micro=HaarMicroPath(3,base)
        self.down1=nn.Sequential(BlurPool2D(base),nn.Conv2d(base,base,3,stride=2,padding=1),nn.SiLU())
        self.down2=nn.Sequential(BlurPool2D(base*2),nn.Conv2d(base*2,base*2,3,stride=2,padding=1),nn.SiLU())
        self.down3=nn.Sequential(BlurPool2D(base*2),nn.Conv2d(base*2,base*2,3,stride=2,padding=1),nn.SiLU())
        self.attn=LocalAttention2D(base*2,window=4,heads=4)
        self.to_moments=nn.Conv2d(base*2,channels*2,3,padding=1)
        self.from_latent=nn.Conv2d(channels,base*2,3,padding=1)
        self.decoder=nn.ModuleList([
            nn.Conv2d(base*2,base*2,3,padding=1),
            nn.Conv2d(base*2,base,3,padding=1),
            nn.Conv2d(base,base,3,padding=1)])
        self.out_rgb=nn.Conv2d(base,3,3,padding=1)
        self.out_uncertainty=nn.Conv2d(base,3,3,padding=1)
        self.encoder_context=nn.Linear(context_dim,base*2)

    def encode(self,x):
        if x.ndim!=4 or x.shape[1]!=3 or not torch.isfinite(x).all():
            raise ContractError('finite NCHW linear RGB tensor required')
        h,w=x.shape[-2:];x=F.pad(x,(0,(-w)%8,0,(-h)%8),mode='replicate')
        ctx=self.context(x)
        macro=self.down1(F.silu(self.stem(x)))
        micro=F.silu(self.micro(x))
        h=self.down3(self.down2(torch.cat((macro,micro),dim=1)))
        h=self.attn(h+self.encoder_context(ctx).unsqueeze(-1).unsqueeze(-1))
        return DiagonalGaussian(self.to_moments(h)),ctx

    def decode(self,z,context=None,return_uncertainty=False):
        if z.ndim!=4 or z.shape[1]!=self.from_latent.in_channels or not torch.isfinite(z).all():
            raise ContractError('invalid latent tensor')
        x=self.from_latent(z)
        if context is not None:
            if context.shape!=(z.shape[0],self.encoder_context.in_features):
                raise ContractError('invalid global context')
            x=x+self.encoder_context(context).unsqueeze(-1).unsqueeze(-1)
        x=self.attn(x)
        for op in self.decoder:
            x=F.silu(op(F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)))
        image=self.out_rgb(x).float()  # intentionally NO sigmoid/clipping
        if return_uncertainty:
            return image,self.out_uncertainty(x).float().clamp(-10,6).exp()
        return image


@dataclass
class CausalStreamState:
    model_id:str
    history:torch.Tensor|None=None
    def reset(self):self.history=None

class CausalConv3D(nn.Module):
    """Causal conv with per-stream state; temporal stride is 1 in this primitive."""
    def __init__(self,in_channels:int,out_channels:int,kernel:int=3,model_id:str='aether-vae-temporal'):
        super().__init__()
        if kernel<1 or kernel%2==0:raise ValueError('odd temporal kernel required')
        self.kernel=kernel;self.model_id=model_id
        self.conv=nn.Conv3d(in_channels,out_channels,(kernel,3,3),padding=(0,1,1))

    def forward(self,x,state:CausalStreamState|None=None):
        if x.ndim!=5 or x.shape[2]<1:raise ContractError('NCTHW frames required')
        if state is not None and state.model_id!=self.model_id:
            raise ContractError('temporal state belongs to a different model')
        if self.kernel==1:return self.conv(x)
        if state is not None and state.history is not None:
            hist=state.history.to(device=x.device,dtype=x.dtype)
            if hist.shape[:2]!=x.shape[:2] or hist.shape[3:]!=x.shape[3:] or hist.shape[2]!=self.kernel-1:
                raise ContractError('invalid temporal history shape')
        else:
            hist=x[:,:,:1].expand(-1,-1,self.kernel-1,-1,-1)
        padded=torch.cat((hist,x),dim=2)
        if state is not None:state.history=padded[:,:,-(self.kernel-1):].detach().clone()
        return self.conv(padded)
