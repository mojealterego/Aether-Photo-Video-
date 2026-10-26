"""AETHERIS OmniFlow Ω: deterministic, non-executable engine catalog.

SYS-AETHERIS-OMNIFLOW-016, OmegaNode ABI v1.1. This registry stores
versioned declarations and dependency order. It NEVER imports or executes
third-party code; only the trusted runtime may bind a verified T0 operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping
from types import MappingProxyType

ABI = 'omega.node.abi.v1.1'
SCHEMA = 'omniflow.catalog.v1'
_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,127}\Z')
_SEMVER = re.compile(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z')
_SHA = re.compile(r'sha256:[a-f0-9]{64}\Z')
_PORT = re.compile(r'[A-Z][A-Z0-9_:.]{0,127}\Z')
_CAPABILITY = re.compile(r'[a-z][a-z0-9_.-]{0,127}\Z')
_ALLOWED_MODES = frozenset(('PURE_DETERMINISTIC','PURE_SEEDED','IMPURE_IDEMPOTENT',
                            'EXTERNAL_SIDE_EFFECT','HUMAN_APPROVAL'))
_ALLOWED_TRUST = frozenset(('T0_CORE_TRUSTED','T1_VERIFIED_PLUGIN','T2_UNTRUSTED_OMEGANODE'))
_MAX_CATALOG_BYTES = 4 * (1 << 20)


class RegistryError(ValueError):
    """Invalid, conflicting, or non-admissible engine declaration."""


class EngineStatus(str, Enum):
    PLANNED = 'PLANNED'
    ON_HOLD = 'ON_HOLD'
    EXPERIMENTAL = 'EXPERIMENTAL'
    VERIFIED = 'VERIFIED'
    REVOKED = 'REVOKED'


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise RegistryError(f'invalid {label}')
    return value


def _version(value: Any) -> str:
    if not isinstance(value, str) or not _SEMVER.fullmatch(value):
        raise RegistryError('invalid exact semver')
    return value


def _capability(value: Any) -> str:
    if not isinstance(value, str) or not _CAPABILITY.fullmatch(value):
        raise RegistryError('invalid capability')
    return value


def _strings(values: Any, label: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or len(values) > 128:
        raise RegistryError(f'invalid {label}')
    output = tuple(_name(item, label) if label != 'capability' else _capability(item) for item in values)
    if len(set(output)) != len(output):
        raise RegistryError(f'duplicate {label}')
    return output


def _ports(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or len(value) > 128:
        raise RegistryError(f'invalid {label}')
    result: dict[str, str] = {}
    for key, semantic_type in value.items():
        name = _name(key, label)
        if not isinstance(semantic_type, str) or not _PORT.fullmatch(semantic_type):
            raise RegistryError(f'invalid semantic type in {label}')
        result[name] = semantic_type
    return result


@dataclass(frozen=True)
class EngineDescriptor:
    op_id: str
    op_version: str
    abi_version: str
    provider: str
    family: str
    trust_class: str
    mode: str
    state: EngineStatus
    provenance_sha256: str
    license_id: str
    dependencies: tuple[tuple[str, str], ...]
    input_ports: dict[str, str]
    output_ports: dict[str, str]
    capabilities: tuple[str, ...]
    test_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        _name(self.op_id, 'op_id')
        _version(self.op_version)
        if self.abi_version != ABI:
            raise RegistryError('incompatible OmegaNode ABI')
        _name(self.provider, 'provider')
        _name(self.family, 'family')
        _name(self.license_id, 'license_id')
        if self.trust_class not in _ALLOWED_TRUST:
            raise RegistryError('invalid trust class')
        if self.mode not in _ALLOWED_MODES:
            raise RegistryError('invalid node mode')
        if not isinstance(self.state, EngineStatus):
            raise RegistryError('invalid engine state')
        if not isinstance(self.provenance_sha256, str) or not _SHA.fullmatch(self.provenance_sha256):
            raise RegistryError('invalid provenance SHA-256')
        if not isinstance(self.dependencies, tuple) or len(self.dependencies) > 128:
            raise RegistryError('invalid dependencies')
        for item in self.dependencies:
            if not isinstance(item, tuple) or len(item) != 2:
                raise RegistryError('invalid dependency pair')
            _name(item[0], 'dependency operator')
            _version(item[1])
        if len(set(self.dependencies)) != len(self.dependencies):
            raise RegistryError('duplicate dependency')
        object.__setattr__(self, 'input_ports', MappingProxyType(_ports(self.input_ports, 'input_ports')))
        object.__setattr__(self, 'output_ports', MappingProxyType(_ports(self.output_ports, 'output_ports')))
        _strings(self.capabilities, 'capability')
        if not isinstance(self.test_evidence, tuple) or len(self.test_evidence) > 128:
            raise RegistryError('invalid test evidence')
        for item in self.test_evidence:
            if not isinstance(item, str) or not (1 <= len(item) <= 512):
                raise RegistryError('invalid test evidence reference')
        if self.state is EngineStatus.VERIFIED and not self.test_evidence:
            raise RegistryError('verified engines require evidence references')

    def to_dict(self) -> dict[str, Any]:
        return {
            'op_id':self.op_id,'op_version':self.op_version,
            'abi_version':self.abi_version,'provider':self.provider,
            'family':self.family,'trust_class':self.trust_class,
            'mode':self.mode,'state':self.state.value,
            'provenance_sha256':self.provenance_sha256,
            'license_id':self.license_id,
            'dependencies':[list(dep) for dep in self.dependencies],
            'input_ports':dict(self.input_ports),'output_ports':dict(self.output_ports),
            'capabilities':list(self.capabilities),'test_evidence':list(self.test_evidence),
        }

    @classmethod
    def from_mapping(cls, obj: Mapping[str, Any]) -> 'EngineDescriptor':
        if not isinstance(obj, dict):
            raise RegistryError('descriptor must be an object')
        required=set(cls.__dataclass_fields__)
        if set(obj) != required:
            raise RegistryError('missing or unknown descriptor fields')
        try:
            state=EngineStatus(obj['state'])
            if not isinstance(obj['dependencies'], list):
                raise RegistryError('dependencies must be a list')
            deps=[]
            for item in obj['dependencies']:
                if not isinstance(item, list) or len(item)!=2:
                    raise RegistryError('dependency must be [operator, exact version]')
                deps.append(tuple(item))
            if not isinstance(obj['test_evidence'], list):
                raise RegistryError('test_evidence must be a list')
            return cls(
                op_id=obj['op_id'],op_version=obj['op_version'],
                abi_version=obj['abi_version'],provider=obj['provider'],
                family=obj['family'],trust_class=obj['trust_class'],mode=obj['mode'],
                state=state,provenance_sha256=obj['provenance_sha256'],
                license_id=obj['license_id'],dependencies=tuple(deps),
                input_ports=_ports(obj['input_ports'],'input_ports'),
                output_ports=_ports(obj['output_ports'],'output_ports'),
                capabilities=_strings(obj['capabilities'],'capability'),
                test_evidence=tuple(obj['test_evidence']),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, RegistryError):
                raise
            raise RegistryError('invalid descriptor contents') from exc

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(),sort_keys=True,separators=(',',':'),ensure_ascii=True,
                          allow_nan=False)

    def digest(self) -> str:
        return 'sha256:'+hashlib.sha256(self.canonical_json().encode('utf-8')).hexdigest()


class Registry:
    """Deterministic catalog without imports, eval or executable entry points."""
    def __init__(self) -> None:
        self._entries: dict[tuple[str,str],EngineDescriptor] = {}

    def register(self, descriptor: EngineDescriptor) -> None:
        if not isinstance(descriptor,EngineDescriptor):
            raise RegistryError('expected validated descriptor')
        key=(descriptor.op_id,descriptor.op_version)
        previous=self._entries.get(key)
        if previous is not None and previous.digest()!=descriptor.digest():
            raise RegistryError('conflicting descriptor for the same exact version')
        self._entries[key]=descriptor

    def list_all(self) -> tuple[EngineDescriptor,...]:
        return tuple(self._entries[k] for k in sorted(self._entries))

    def resolve(self,op_id:str,op_version:str,*,require_verified:bool=False) -> EngineDescriptor:
        descriptor=self._entries.get((_name(op_id,'op_id'),_version(op_version)))
        if descriptor is None:
            raise RegistryError('unknown operator or exact version')
        if require_verified and descriptor.state is not EngineStatus.VERIFIED:
            raise RegistryError('engine not eligible for production activation')
        return descriptor

    def plan(self, root:tuple[str,str],*,require_verified:bool=False) -> tuple[EngineDescriptor,...]:
        if not isinstance(root,tuple) or len(root)!=2:
            raise RegistryError('invalid root')
        visited:set[tuple[str,str]]=set()
        active:set[tuple[str,str]]=set()
        order:list[EngineDescriptor]=[]

        def visit(key:tuple[str,str]) -> None:
            if key in active:
                raise RegistryError('dependency cycle')
            if key in visited:
                return
            if key not in self._entries:
                raise RegistryError('missing dependency or operator')
            node=self.resolve(key[0],key[1],require_verified=require_verified)
            active.add(key)
            for dependency in sorted(node.dependencies):
                visit(dependency)
            active.remove(key)
            visited.add(key)
            order.append(node)

        visit(root)
        return tuple(order)

    def validate_link(self, source:tuple[str,str], source_port:str,
                      target:tuple[str,str], target_port:str,*,
                      require_verified:bool=False) -> None:
        if not (isinstance(source,tuple) and len(source)==2 and
                isinstance(target,tuple) and len(target)==2):
            raise RegistryError('malformed link endpoint')
        a=self.resolve(*source,require_verified=require_verified)
        b=self.resolve(*target,require_verified=require_verified)
        x=a.output_ports.get(_name(source_port,'output port'))
        y=b.input_ports.get(_name(target_port,'input port'))
        if x is None or y is None:
            raise RegistryError('missing port')
        if x!=y:
            raise RegistryError('semantic type mismatch; explicit adapter required')

    def find_capability(self,capability:str,*,require_verified:bool=False) -> tuple[EngineDescriptor,...]:
        _capability(capability)
        return tuple(d for d in self.list_all() if capability in d.capabilities and
                     (not require_verified or d.state is EngineStatus.VERIFIED))

    def to_json(self) -> str:
        return json.dumps({'version':SCHEMA,'entries':[x.to_dict() for x in self.list_all()]},
                          separators=(',',':'),sort_keys=True,ensure_ascii=True,allow_nan=False)

    @classmethod
    def from_json(cls,payload:str) -> 'Registry':
        if not isinstance(payload,str) or len(payload.encode('utf-8'))>_MAX_CATALOG_BYTES:
            raise RegistryError('catalog is not a bounded UTF-8 document')
        try:
            data=json.loads(payload)
        except (TypeError,ValueError) as exc:
            raise RegistryError('invalid JSON catalog') from exc
        if not isinstance(data,dict) or set(data)!= {'version','entries'} or data['version']!=SCHEMA:
            raise RegistryError('unsupported catalog version')
        if not isinstance(data['entries'],list) or len(data['entries'])>10000:
            raise RegistryError('invalid catalog entry count')
        registry=cls()
        for item in data['entries']:
            registry.register(EngineDescriptor.from_mapping(item))
        return registry
