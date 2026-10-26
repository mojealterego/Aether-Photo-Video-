"""SYS-AETHERIS-VISION-VAE-017 phase TR-0: training data eligibility gate."""
from __future__ import annotations
from dataclasses import dataclass
from .contracts import ContractError, _HASH

class GovernanceError(ContractError):pass

@dataclass(frozen=True)
class SampleRecord:
    sample_id:str
    immutable_content_sha256:str
    split_id:str
    acquisition_license_id:str
    permitted_training_scopes:tuple[str,...]
    source_format:str
    bit_depth:int
    color_profile:str
    scene_or_session_id:str
    subject_group_id:str
    consent_status:str

def verify_dataset(records:list[SampleRecord],*,required_scope:str='vae_training')->bool:
    if not records or not required_scope:raise GovernanceError('empty manifest/scope')
    sessions={};subjects={};samples=set();digests={}
    for rec in records:
        if not isinstance(rec,SampleRecord) or not rec.sample_id or rec.sample_id in samples:
            raise GovernanceError('duplicate or invalid sample')
        samples.add(rec.sample_id)
        if not _HASH.fullmatch(rec.immutable_content_sha256):
            raise GovernanceError('invalid source SHA-256')
        if (not rec.acquisition_license_id or rec.consent_status not in ('verified_granted','synthetic_exempt')
            or required_scope not in rec.permitted_training_scopes):
            raise GovernanceError('missing training rights or consent')
        if not rec.color_profile or rec.bit_depth not in (8,10,12,14,16,32):
            raise GovernanceError('unknown source color or bit depth')
        if rec.split_id not in ('train','val','test') or not rec.scene_or_session_id:
            raise GovernanceError('invalid data split or session')
        prev=digests.setdefault(rec.immutable_content_sha256,rec.split_id)
        if prev!=rec.split_id:raise GovernanceError('same content across train/val/test')
        prev=sessions.setdefault(rec.scene_or_session_id,rec.split_id)
        if prev!=rec.split_id:raise GovernanceError('same session across train/val/test')
        if rec.subject_group_id:
            prev=subjects.setdefault(rec.subject_group_id,rec.split_id)
            if prev!=rec.split_id:raise GovernanceError('same subject across train/val/test')
    return True
