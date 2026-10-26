"""AETHERIS Ω-TC16: anchor-preserving causal group codec (untrained topology).

f_s=8, f_t=4 after first anchor. A latent group for source frames 1..4
is emitted only after all four frames arrive (3-frame group lookahead).
At EOS an incomplete group has no latent; missing tail is explicitly
predicted by last-frame hold. This is lossy, never archival or bit-exact.
Streaming states are local to one model and one clip, not thread safe.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

class TemporalContractError(ValueError): pass

@dataclass
class EncoderState:
    model_id:str
    pending:torch.Tensor|None=None
    history:torch.Tensor|None=None
    first:bool=True
    frame_index:int=0
    shape:tuple[int,int,int]|None=None
    finished:bool=False
    def reset(self):
        self.pending=None;self.history=None;self.first=True
        self.frame_index=0;self.shape=None;self.finished=False

@dataclass
class DecoderState:
    model_id:str
    history:torch.Tensor|None=None
    last_frame:torch.Tensor|None=None
    first:bool=True
    emitted:int=0
    shape:tuple[int,int,int]|None=None
    finished:bool=False
    def reset(self):
        self.history=None;self.last_frame=None;self.first=True;self.emitted=0
        self.shape=None;self.finished=False

class TemporalPosterior:
    def __init__(self,moments:torch.Tensor):
        if moments.ndim!=5 or moments.shape[1]%2 or not torch.isfinite(moments).all():
            raise TemporalContractError('finite NCTHW posterior moments required')
        self.mean,self.logvar=moments.float().chunk(2,dim=1)
        self.logvar=self.logvar.clamp(-30,20)
    def mode(self):return self.mean
    def sample(self,seed:int|None=None):
        gen=torch.Generator(device=self.mean.device)
        if seed is not None:gen.manual_seed(int(seed))
        noise=torch.randn(self.mean.shape,device=self.mean.device,dtype=self.mean.dtype,
                          generator=gen if seed is not None else None)
        return self.mean+torch.exp(.5*self.logvar)*noise
    def kl(self):
        return .5*(self.mean.square()+self.logvar.exp()-1-self.logvar).mean()

class _GroupCausalConv(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.conv=nn.Conv3d(in_channels,out_channels,(3,3,3),padding=(0,1,1))
    def forward(self,x:torch.Tensor,history:torch.Tensor|None):
        if x.shape[2]<1:raise TemporalContractError('group count must be positive')
        if history is None:
            history=x[:,:,:1].expand(-1,-1,2,-1,-1)
        if history.shape!=(x.shape[0],x.shape[1],2,x.shape[3],x.shape[4]):
            raise TemporalContractError('group history geometry mismatch')
        padded=torch.cat((history,x),dim=2)
        return self.conv(padded),padded[:,:,-2:].detach().clone()

class TemporalVAE16(nn.Module):
    """Trainable TC16 encoder/decoder; no pretrained or approved weights included.

    encode_chunk never pads at chunk boundaries. On EOS the incomplete
    source group is omitted (spec: Tz=1+floor((T-1)/4)). The decoder's
    final missing 1-3 frames are explicitly a last-frame prediction.
    """
    def __init__(self,base_channels:int=48,latent_channels:int=16,model_id:str='aethervae-tc16-v1'):
        super().__init__()
        if type(base_channels)!=int or base_channels<4 or type(latent_channels)!=int or latent_channels<1 or not model_id:
            raise TemporalContractError('invalid TC16 architecture')
        self.base_channels=base_channels
        self.latent_channels=latent_channels
        self.model_id=model_id
        c=base_channels
        self.stem=nn.Sequential(
            nn.Conv2d(3,c,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(c,c,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(c,c,3,stride=2,padding=1),nn.SiLU())
        self.anchor_to_moments=nn.Conv2d(c,latent_channels*2,1)
        self.group_reduce=nn.Conv3d(c,c,(4,1,1),stride=(4,1,1))
        self.enc_causal=_GroupCausalConv(c,c)
        self.group_to_moments=nn.Conv3d(c,latent_channels*2,1)
        self.anchor_from_latent=nn.Conv2d(latent_channels,c,1)
        self.dec_causal=_GroupCausalConv(latent_channels,c)
        self.group_expand=nn.Conv3d(c,c*4,1)
        self.decode_frames=nn.Sequential(
            nn.Conv2d(c,c,3,padding=1),nn.SiLU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(c,c,3,padding=1),nn.SiLU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(c,c,3,padding=1),nn.SiLU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False))
        self.out_rgb=nn.Conv2d(c,3,3,padding=1)

    def _spatial_stem(self,x):
        b,c,t,h,w=x.shape
        y=self.stem(x.permute(0,2,1,3,4).reshape(b*t,c,h,w))
        _,ch,hh,ww=y.shape
        return y.reshape(b,t,ch,hh,ww).permute(0,2,1,3,4)

    def _spatial_decode(self,x,output_hw):
        b,c,t,h,w=x.shape
        if t==0:return x.new_empty((b,3,0,*(output_hw or (h*8,w*8))))
        z=x.permute(0,2,1,3,4).reshape(b*t,c,h,w)
        z=self.out_rgb(self.decode_frames(z)).float()
        z=z.reshape(b,t,3,z.shape[-2],z.shape[-1]).permute(0,2,1,3,4)
        if output_hw is not None:z=z[:,:,:,:output_hw[0],:output_hw[1]]
        return z

    def encode_chunk(self,frames:torch.Tensor,state:EncoderState|None=None,*,scene_cut:bool=False,finalize:bool=False):
        if frames.ndim!=5 or frames.shape[1]!=3 or frames.shape[2]<1 or not torch.isfinite(frames).all():
            raise TemporalContractError('finite RGB NCTHW frames required')
        if state is None:state=EncoderState(self.model_id)
        if state.model_id!=self.model_id:raise TemporalContractError('foreign model encoder state')
        if scene_cut:state.reset()
        if state.finished:raise TemporalContractError('encoder state already finalized')
        b,_,t,h,w=frames.shape
        if state.shape is None:state.shape=(b,h,w)
        elif state.shape!=(b,h,w):raise TemporalContractError('source frame shape changed within scene')
        frames=F.pad(frames,(0,(-w)%8,0,(-h)%8,0,0),mode='replicate')
        features=self._spatial_stem(frames)
        outputs=[]
        if state.first:
            outputs.append(self.anchor_to_moments(features[:,:,0]).unsqueeze(2))
            features=features[:,:,1:]
            state.first=False
        if state.pending is not None:
            features=torch.cat((state.pending,features),dim=2)
        n=features.shape[2]//4
        if n:
            consumed=features[:,:,:n*4]
            grouped=self.group_reduce(consumed)
            mixed,state.history=self.enc_causal(grouped,state.history)
            outputs.append(self.group_to_moments(F.silu(mixed)))
        state.pending=features[:,:,n*4:]
        state.frame_index+=t
        if finalize:
            state.pending=None
            state.finished=True
        if not outputs:
            _,ch,_,hh,ww=features.shape
            moments=features.new_empty((b,self.latent_channels*2,0,hh,ww))
        else:moments=torch.cat(outputs,dim=2)
        return TemporalPosterior(moments),state

    def decode_chunk(self,z:torch.Tensor,state:DecoderState|None=None,*,scene_cut:bool=False,
                     target_frames:int|None=None,output_hw:tuple[int,int]|None=None,finalize:bool=False):
        if z.ndim!=5 or z.shape[1]!=self.latent_channels or not torch.isfinite(z).all():
            raise TemporalContractError('finite TC16 latent NCTHW required')
        if state is None:state=DecoderState(self.model_id)
        if state.model_id!=self.model_id:raise TemporalContractError('foreign model decoder state')
        if scene_cut:state.reset()
        if state.finished:raise TemporalContractError('decoder state already finalized')
        b,_,n,hh,ww=z.shape
        if state.shape is None:state.shape=(b,hh,ww)
        elif state.shape!=(b,hh,ww):raise TemporalContractError('latent spatial geometry changed')
        if output_hw is not None and (len(output_hw)!=2 or min(output_hw)<=0 or output_hw[0]>hh*8 or output_hw[1]>ww*8):
            raise TemporalContractError('invalid output spatial geometry')
        if target_frames is not None and (type(target_frames)!=int or target_frames<1 or not finalize):
            raise TemporalContractError('target_frames requires final chunk')
        groups=[]
        if state.first and n:
            groups.append(self.anchor_from_latent(z[:,:,0]).unsqueeze(2))
            z=z[:,:,1:];state.first=False
        if z.shape[2]:
            features,state.history=self.dec_causal(z,state.history)
            expanded=self.group_expand(F.silu(features))
            nb,c4,k,h,w=expanded.shape
            expanded=expanded.reshape(nb,4,self.base_channels,k,h,w).permute(0,2,3,1,4,5)
            groups.append(expanded.reshape(nb,self.base_channels,k*4,h,w))
        if groups:recon=self._spatial_decode(torch.cat(groups,dim=2),output_hw)
        else:recon=z.new_empty((b,3,0,*(output_hw or (hh*8,ww*8))))
        state.emitted+=recon.shape[2]
        if recon.shape[2]:state.last_frame=recon[:,:,-1:].detach().clone()
        if finalize:
            if target_frames is not None:
                extra=target_frames-state.emitted
                if extra<0 or extra>3 or (extra and state.last_frame is None):
                    raise TemporalContractError('invalid EOS tail; cannot predict beyond 3 frames')
                if extra:
                    recon=torch.cat((recon,state.last_frame.expand(-1,-1,extra,-1,-1)),dim=2)
                    state.emitted+=extra
            state.finished=True
        return recon,state

    def encode(self,x):
        posterior,_=self.encode_chunk(x,finalize=True)
        return posterior

    def decode(self,z,*,target_frames:int,output_hw:tuple[int,int]|None=None):
        frames,_=self.decode_chunk(z,finalize=True,target_frames=target_frames,output_hw=output_hw)
        return frames

    def forward(self,x):
        posterior=self.encode(x)
        reconstruction=self.decode(posterior.mode(),target_frames=x.shape[2],output_hw=x.shape[-2:])
        return reconstruction,posterior
