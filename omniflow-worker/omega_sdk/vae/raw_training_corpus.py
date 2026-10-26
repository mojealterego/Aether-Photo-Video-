"""Explicit-authorization metadata planner for RAW photo model training.

Never infers a training license from a file upload, URL or shared-folder link.
Only a specifically user-granted corpus may pass; JPEG/MP4 are excluded.
Folder grouping mitigates session leakage, but global perceptual
near-duplicate checking remains mandatory before production promotion.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import re

class CorpusAdmissionError(ValueError): pass

_TOKEN=re.compile(r'[A-Za-z0-9_-]{4,128}\Z')
_SHA=re.compile(r'[a-f0-9]{64}\Z')

@dataclass(frozen=True)
class TrainingGrant:
    root_folder_id: str
    authorized: bool
    purpose: str
    source: str

@dataclass(frozen=True)
class CorpusSource:
    file_id: str
    folder_id: str
    filename: str
    byte_size: int
    sha256: str|None = None

def build_split_plan(entries: list[CorpusSource], grant: TrainingGrant,
                     *, root_folder_id: str|None = None,
                     seed: str='AETHERIS-OMEGA-RAW-SPLIT-v1') -> dict:
    if (not isinstance(grant, TrainingGrant) or grant.authorized is not True
        or grant.purpose != 'PHOTO_VAE_TRAINING'
        or grant.source != 'USER_EXPLICIT_CHAT_CONSENT'):
        raise CorpusAdmissionError('explicit training authorization for photo VAE required')
    if not _TOKEN.fullmatch(grant.root_folder_id) or not isinstance(seed,str) or not seed:
        raise CorpusAdmissionError('malformed grant/split seed')
    if root_folder_id is not None and grant.root_folder_id != root_folder_id:
        raise CorpusAdmissionError('grant scope mismatch')
    if not isinstance(entries, list) or not entries or len(entries)>100_000:
        raise CorpusAdmissionError('empty or excessively large source catalog')
    ids=set(); hashes=set(); grouped=defaultdict(list)
    bytes_total=0;missing=0
    for e in entries:
        if (not isinstance(e,CorpusSource) or not isinstance(e.file_id,str)
            or not _TOKEN.fullmatch(e.file_id) or e.file_id in ids):
            raise CorpusAdmissionError('invalid or duplicate Drive file ID')
        ids.add(e.file_id)
        if not isinstance(e.folder_id,str) or not _TOKEN.fullmatch(e.folder_id):
            raise CorpusAdmissionError('invalid parent folder group')
        if (not isinstance(e.filename,str) or '\\' in e.filename or '/' in e.filename
            or not e.filename.upper().endswith('.NEF')):
            raise CorpusAdmissionError('NEF originals only: JPEG/video have separate grants')
        if type(e.byte_size) is not int or not 131_072<=e.byte_size<=16*(1<<30):
            raise CorpusAdmissionError('invalid original RAW byte length')
        if e.sha256 is None:missing+=1
        elif not isinstance(e.sha256,str) or not _SHA.fullmatch(e.sha256) or e.sha256 in hashes:
            raise CorpusAdmissionError('invalid or duplicate source SHA-256')
        else:hashes.add(e.sha256)
        grouped[e.folder_id].append(e)
        bytes_total+=e.byte_size
    folders=sorted(grouped,key=lambda folder: (
        hashlib.sha256((seed+':'+folder).encode()).hexdigest(),folder))
    n=len(folders)
    withheld=max(1,int(round(n*.1))) if n>=3 else 0
    val=set(folders[:withheld]);test=set(folders[withheld:2*withheld])
    if n>=3 and len(val)+len(test)>=n:
        raise CorpusAdmissionError('not enough training groups for split')
    def kind(fid):return 'validation' if fid in val else ('test' if fid in test else 'train')
    indexed=[{'file_id':e.file_id,'folder_id':e.folder_id,'filename':e.filename,
              'byte_size':e.byte_size,'sha256':e.sha256,'split':kind(e.folder_id)}
             for e in sorted(entries,key=lambda x:(x.folder_id,x.file_id))]
    split_counts={key:sum(1 for e in indexed if e['split']==key)
                  for key in ('train','validation','test')}
    return {'schema':'aether.raw_training_corpus/1.0',
            'scope_root_folder_id':grant.root_folder_id,
            'user_explicitly_authorized_training':True,
            'purpose':grant.purpose,'file_count':len(indexed),'group_count':n,
            'source_bytes':bytes_total,'missing_file_sha256':missing,
            'status':'PROVISIONAL_METADATA_ONLY' if missing else 'PENDING_CROSS_FOLDER_VISUAL_DEDUPLICATION',
            'ready_for_production_training':False,
            'split_counts':split_counts,'split_seed':seed,
            'split_policy':'GROUP_BY_PARENT_FOLDER_THEN_DETERMINISTIC_HASH',
            'jpg_and_video_excluded':True,
            'near_duplicate_leakage_verified':False,
            'files':indexed}
