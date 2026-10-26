"""Ω-TC16 safe, bounded synthetic-only training smoke, NOT TR-1..TR-6.

No dataset, checkpoint, or training metric from this module qualifies
for signed model release. Real training requires governed video data,
held-out tests, GPU training, and independent promotion.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import torch
from safetensors.torch import save_file
from .governance import SampleRecord,verify_dataset,GovernanceError
from .temporal_tc16 import TemporalVAE16

class TrainingGateError(ValueError):pass

def train_smoke(model:TemporalVAE16,batches:list[tuple[SampleRecord,torch.Tensor]],
                output:Path,*,steps:int=1,lr:float=1e-3)->dict:
    if not isinstance(model,TemporalVAE16) or not 1<=steps<=100 or not 0<lr<=.01:
        raise TrainingGateError('bounded TC16 smoke settings required')
    if not batches or len(batches)>64:
        raise TrainingGateError('bounded synthetic samples required')
    records=[record for record,_ in batches]
    try:verify_dataset(records)
    except GovernanceError as exc:raise TrainingGateError('dataset governance denied') from exc
    if any(record.consent_status!='synthetic_exempt' or record.source_format!='SYNTHETIC' or record.split_id!='train' for record in records):
        raise TrainingGateError('smoke accepts synthetic train samples only; real release requires TR-0..TR-6')
    for _,x in batches:
        if x.ndim!=5 or x.shape[1]!=3 or x.shape[2]<5 or x.shape[2]>33 or max(x.shape[-2:])>256 or not torch.isfinite(x).all():
            raise TrainingGateError('bounded finite video tensor required')
    output=Path(output)
    if output.exists() or output.suffix!='.safetensors':raise TrainingGateError('immutable .safetensors destination required')
    tmp=output.with_name(output.name+'.partial')
    if tmp.exists():raise TrainingGateError('unfinished checkpoint exists')
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    losses=[]
    try:
        for i in range(steps):
            _,x=batches[i%len(batches)]
            optimizer.zero_grad(set_to_none=True)
            recon,posterior=model(x)
            loss=(recon-x).square().mean()+1e-6*posterior.kl()
            if not torch.isfinite(loss):raise TrainingGateError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(tmp))
        with tmp.open('rb') as fd:os.fsync(fd.fileno())
        h=hashlib.sha256()
        with tmp.open('rb') as fd:
            for chunk in iter(lambda:fd.read(1<<20),b''):h.update(chunk)
        os.replace(tmp,output)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return {'status':'SMOKE_UNAPPROVED','production_approved':False,'steps':steps,
            'initial_loss':losses[0],'final_loss':losses[-1],
            'weights_sha256':'sha256:'+h.hexdigest(),
            'held_out_validation':False,'gpu_e2e':False,
            'note':'synthetic training smoke; never sign/promote as trained validated release'}
