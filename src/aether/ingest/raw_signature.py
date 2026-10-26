"""Bounded signature gate for Canon CIFF/CRW, Canon CR2 and DNG.

Extensions are NOT authoritative. Headerless camera/CHDK sensor dumps are
recognized as unverified candidates, never automatically admitted.
This is format sniffing, NOT provenance, demosaic or color certification.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import struct

class RawSignatureError(ValueError):
    pass

class RawContainerKind(str, Enum):
    CANON_CR2 = 'CANON_CR2'
    CANON_CIFF_CRW = 'CANON_CIFF_CRW'
    DNG = 'DNG'
    UNCLASSIFIED_TIFF = 'UNCLASSIFIED_TIFF'
    HEADERLESS_SENSOR_DUMP = 'HEADERLESS_SENSOR_DUMP_UNVERIFIED'
    UNKNOWN = 'UNKNOWN'

@dataclass(frozen=True)
class RawContainerInfo:
    kind: RawContainerKind
    filename_suffix: str
    standard_container: bool
    extension_mismatch: bool
    admitted_by_signature: bool

_TIFF_LE = b'II*\x00'
_TIFF_BE = b'MM\x00*'
_MAX_PREFIX = 1 << 16

def _has_dng_version(prefix: bytes, *, little_endian: bool) -> bool:
    """Read DNGVersion tag 50706 from first IFD, strictly within 64 KiB."""
    if len(prefix) < 10:
        return False
    endian = '<' if little_endian else '>'
    offset = struct.unpack_from(endian + 'I', prefix, 4)[0]
    if offset < 8 or offset > len(prefix) - 2:
        return False
    count = struct.unpack_from(endian + 'H', prefix, offset)[0]
    if count > 4096 or count * 12 > len(prefix) - offset - 2:
        return False
    for index in range(count):
        pos = offset + 2 + index * 12
        tag, dtype, number = struct.unpack_from(endian + 'HHI', prefix, pos)
        if tag == 50706:
            return dtype == 1 and number == 4 and len(prefix) >= pos + 12
    return False

def classify_raw_container(source: str | Path) -> RawContainerInfo:
    """Read only 64 KiB to classify potential Canon RAW source safely."""
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise RawSignatureError('source must be an existing regular non-symlink')
    if path.stat().st_size < 8:
        raise RawSignatureError('truncated source')
    suffix = path.suffix.upper()
    with path.open('rb') as fd:
        header = fd.read(_MAX_PREFIX)
    kind = RawContainerKind.UNKNOWN
    standard = False
    if header.startswith(_TIFF_LE) or header.startswith(_TIFF_BE):
        if len(header) >= 12 and header[8:12] == b'CR\x02\x00':
            kind, standard = RawContainerKind.CANON_CR2, True
        elif _has_dng_version(header, little_endian=header.startswith(_TIFF_LE)):
            kind, standard = RawContainerKind.DNG, True
        else:
            kind = RawContainerKind.UNCLASSIFIED_TIFF
    elif len(header) >= 14 and header[:6] == b'II\x1a\x00\x00\x00' and header[6:14] == b'HEAPCCDR':
        kind, standard = RawContainerKind.CANON_CIFF_CRW, True
    elif suffix in ('.CRW', '.CR2'):
        kind = RawContainerKind.HEADERLESS_SENSOR_DUMP
    expected_suffix = {
        RawContainerKind.CANON_CR2: '.CR2',
        RawContainerKind.CANON_CIFF_CRW: '.CRW',
        RawContainerKind.DNG: '.DNG',
    }.get(kind)
    mismatched = expected_suffix is not None and expected_suffix != suffix
    return RawContainerInfo(kind, suffix, standard, mismatched,
                            bool(standard and not mismatched))
