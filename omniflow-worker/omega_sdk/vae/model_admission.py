"""Local signed release admission for AETHERIS Ω-VAE model bundles.

Ed25519 is an additive hardened verifier to the source HMAC design.
A signature proves provenance from an allowlisted key, not training quality,
model rights or source-image permissions. No pickle or network used.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
from typing import Mapping

from .contracts import LatentSchema, ColorContract, checked_elements, ContractError

class AdmissionError(ValueError):pass

_SHA=re.compile(r'sha256:[0-9a-f]{64}\Z')
_VERSION=re.compile(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z')
_DTYPE_SIZE={'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'U64':8,'I32':4,'U32':4,
             'I16':2,'U16':2,'I8':1,'U8':1,'BOOL':1}

@dataclass(frozen=True)
class WeightFile:
    role:str
    relative_path:str
    sha256:str

@dataclass(frozen=True)
class BundleManifest:
    schema_version:str
    package_id:str
    package_semver:str
    release_status:str
    runtime_abi:str
    latent_schema_sha256:str
    color_contract_sha256:str
    weights:tuple[WeightFile,...]
    license_id:str
    commercial_use_allowed:bool
    signer_key_id:str
    signature_hex:str

    def canonical_payload(self)->bytes:
        obj=asdict(self)
        obj.pop('signature_hex')
        return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')

def _bad_digest(value):
    return not isinstance(value,str) or not _SHA.fullmatch(value)

def _inspect_safetensors(path:Path)->None:
    """Validate bounded header and exact offsets without materializing weights."""
    max_header=100<<20
    size=path.stat().st_size
    if size<10 or size>8<<30:raise AdmissionError('invalid safetensors file size')
    with path.open('rb') as fd:
        pre=fd.read(8)
        if len(pre)!=8:raise AdmissionError('truncated model header')
        n=struct.unpack('<Q',pre)[0]
        if n<2 or n>max_header or n+8>size:raise AdmissionError('invalid safetensors header length')
        raw=fd.read(n)
    def distinct(items):
        obj={}
        for k,v in items:
            if k in obj:raise AdmissionError('duplicate safetensors field')
            obj[k]=v
        return obj
    def reject_nan(_):raise AdmissionError('nonfinite safetensors JSON')
    try:
        obj=json.loads(raw.decode('utf-8'),object_pairs_hook=distinct,parse_constant=reject_nan)
    except (ValueError,UnicodeDecodeError,TypeError) as exc:
        raise AdmissionError('invalid safetensors JSON') from exc
    if not isinstance(obj,dict):raise AdmissionError('invalid safetensors tensor dictionary')
    intervals=[]
    for name,info in obj.items():
        if name=='__metadata__':
            if not isinstance(info,dict) or any(type(k) is not str or type(v) is not str for k,v in info.items()):
                raise AdmissionError('invalid safetensors metadata')
            continue
        if not isinstance(name,str) or not name or not isinstance(info,dict) or set(info)!= {'dtype','shape','data_offsets'}:
            raise AdmissionError('invalid tensor record')
        dtype=info['dtype']; dims=info['shape']; offsets=info['data_offsets']
        if type(dtype) is not str or dtype not in _DTYPE_SIZE or not isinstance(dims,list) or not isinstance(offsets,list) or len(offsets)!=2:
            raise AdmissionError('invalid tensor shape or dtype')
        try: elements=checked_elements(dims, (8<<30)//_DTYPE_SIZE[dtype]) if dims else 1
        except ContractError as exc:raise AdmissionError('invalid model tensor dimensions') from exc
        a,b=offsets
        if type(a) is not int or type(b) is not int or a<0 or b<a or b>size-n-8 or (b-a)!=elements*_DTYPE_SIZE[dtype]:
            raise AdmissionError('invalid tensor byte range')
        intervals.append((a,b))
    pos=0
    for start,end in sorted(intervals):
        if start!=pos:raise AdmissionError('gap or overlap in weight bytes')
        pos=end
    if pos!=size-n-8:raise AdmissionError('unindexed tensor data bytes')

class BundleAdmission:
    def __init__(self,*,trusted_signers:Mapping[str,bytes]):
        self.signers=dict(trusted_signers)

    def verify(self,manifest:BundleManifest,*,root:Path,latent_schema:LatentSchema,
               color_contract:ColorContract,commercial_use:bool)->dict:
        if not isinstance(manifest,BundleManifest):raise AdmissionError('typed model manifest required')
        if manifest.schema_version!='aethervae.model_bundle/1.0' or manifest.runtime_abi!='omega.node.abi.v1.1':
            raise AdmissionError('model ABI mismatch')
        if not isinstance(manifest.package_id,str) or not manifest.package_id or not _VERSION.fullmatch(manifest.package_semver):
            raise AdmissionError('invalid package identity or version')
        if manifest.release_status!='TRAINED_SIGNED_VALIDATED':raise AdmissionError('untrained or unapproved model release')
        if not isinstance(manifest.license_id,str) or not manifest.license_id or type(manifest.commercial_use_allowed) is not bool:
            raise AdmissionError('missing license contract')
        if commercial_use and not manifest.commercial_use_allowed:raise AdmissionError('commercial use denied')
        if _bad_digest(manifest.latent_schema_sha256) or _bad_digest(manifest.color_contract_sha256):
            raise AdmissionError('missing data contract digest')
        if manifest.latent_schema_sha256!=latent_schema.digest() or manifest.color_contract_sha256!=color_contract.digest():
            raise AdmissionError('latent or color schema mismatch')
        key=self.signers.get(manifest.signer_key_id)
        if key is None or not isinstance(key,bytes) or len(key)!=32:
            raise AdmissionError('unknown signing authority')
        signature=manifest.signature_hex
        if not isinstance(signature,str) or len(signature)!=128 or not re.fullmatch(r'[0-9a-f]{128}',signature):
            raise AdmissionError('missing or invalid release signature')
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
        try:
            Ed25519PublicKey.from_public_bytes(key).verify(bytes.fromhex(signature),manifest.canonical_payload())
        except (InvalidSignature,ValueError,TypeError) as exc:
            raise AdmissionError('release signature verification failed') from exc
        if not isinstance(manifest.weights,tuple) or not 1<=len(manifest.weights)<=128:
            raise AdmissionError('verified model requires pinned weight files')
        root=Path(root).resolve(strict=True)
        roles=set();paths=set()
        for item in manifest.weights:
            if not isinstance(item,WeightFile) or not item.role or item.role in roles or _bad_digest(item.sha256):
                raise AdmissionError('invalid or duplicate weight role/hash')
            roles.add(item.role)
            if not isinstance(item.relative_path,str) or not item.relative_path or '\\' in item.relative_path:
                raise AdmissionError('invalid local weight path')
            rel=Path(item.relative_path)
            if (rel.is_absolute() or any(part in ('..','.') for part in rel.parts)
                or rel.suffix!='.safetensors' or rel in paths):
                raise AdmissionError('path traversal, duplicate or unsafe weight format')
            paths.add(rel)
            candidate=root/rel
            if any(p.is_symlink() for p in [candidate,*list(candidate.parents)[:-1]] if p != root and root in p.parents):
                raise AdmissionError('symlink in model bundle path')
            try:
                resolved=candidate.resolve(strict=True)
                if not resolved.is_relative_to(root) or not stat.S_ISREG(resolved.stat().st_mode):
                    raise AdmissionError('model must remain within the trusted bundle')
                _inspect_safetensors(resolved)
                digest=hashlib.sha256()
                with resolved.open('rb') as fd:
                    for chunk in iter(lambda:fd.read(1<<20),b''):digest.update(chunk)
            except (OSError,ValueError) as exc:
                raise AdmissionError('model weights missing or corrupt') from exc
            if 'sha256:'+digest.hexdigest()!=item.sha256:raise AdmissionError('weight checksum mismatch')
        return {'package_id':manifest.package_id,'version':manifest.package_semver,
                'weight_count':len(manifest.weights),'signer_key_id':manifest.signer_key_id,
                'trusted_signature_verified':True,'model_quality_certified':False}
