"""AETHERIS private corpus intake: read-only Drive metadata and fail-closed splits.

All training data remain blocked pending content SHA256, duplicate/session/
subject grouping, camera color metadata and independent rights review.
This source code contains NO user's private Drive folder IDs or filenames.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

SCHEMA = 'aetheris.omniflow.private_training_corpus/1.0'
_FORMATS = {'NEF': 'CAMERA_RAW', 'DNG': 'CAMERA_RAW', 'RAF': 'CAMERA_RAW',
            'CR2': 'CAMERA_RAW', 'CRW': 'CAMERA_RAW',
            'JPG': 'DERIVED_OR_CAMERA_JPEG', 'JPEG': 'DERIVED_OR_CAMERA_JPEG',
            'MP4': 'VIDEO_UNDETERMINED', 'MOV': 'VIDEO_UNDETERMINED'}
_MD5_RE = re.compile(r'^[a-fA-F0-9]{32}$')


class CorpusGateError(ValueError):
    pass


def _split_hint(group_id: str) -> str:
    """Repeatable 80/10/10 at folder level; NOT a validated final split."""
    value = hashlib.sha256(
        ('aetheris.20261026.group-split.v1/' + group_id).encode()
    ).digest()
    sample = int.from_bytes(value[:8], 'big') % 100
    return 'train' if sample < 80 else ('val' if sample < 90 else 'test')


def plan_from_inventory(inventory: dict, *, consent: dict,
                        approved_roots: dict[str, str]) -> dict:
    """Make a PRIVATE manifest from a six-root, folder-level Drive inventory."""
    if inventory.get('schema') != 'aether.raw_drive_metadata_inventory/1.0':
        raise CorpusGateError('unexpected inventory schema')
    if consent.get('authorization_from_uploader') != 'GRANTED':
        raise CorpusGateError('explicit training authorization missing')
    sources = {item['root']: item for item in inventory['roots']}
    if len(approved_roots) != 6 or set(sources) != set(approved_roots):
        raise CorpusGateError('approved folder roots do not match inventory')
    for code, source in sources.items():
        if source['url'].rstrip('/').split('/')[-1] != approved_roots[code]:
            raise CorpusGateError('Drive folder ID mismatch')
    groups = []
    for entry in inventory['subfolders']:
        root = entry['root']
        if root not in approved_roots:
            raise CorpusGateError('unapproved root')
        group_id = root + '/' + str(entry['subfolder_code'])
        formats = entry['counts']
        if (type(entry['files_min']) is not int or entry['files_min'] < 0 or
                sum(formats.values()) != entry['files_min']):
            raise CorpusGateError('invalid folder counts')
        groups.append({
            'group_id': group_id,
            'source_root': root,
            'folder_code': entry['subfolder_code'],
            'format_counts_min': formats,
            'files_min': entry['files_min'],
            'bytes_min': entry['size_bytes_min'],
            'listing_complete': bool(entry['complete']),
            'split_hint': _split_hint(group_id),
            'final_split': None,
            'trainable': False,
            'hold_reasons': [
                'FILE_RECORDS_NOT_YET_ENUMERATED',
                'SHA256_AND_DEDUP_NOT_VERIFIED',
                'RIGHTS_SUBJECT_COLOR_REVIEW_REQUIRED'
            ] + ([] if entry['complete'] else ['DRIVE_PAGINATION_REQUIRED'])
        })
    count = Counter()
    for group in groups:
        count.update(group['format_counts_min'])
    if dict(count) != inventory['formats_minimum']:
        raise CorpusGateError('root/child format totals mismatch')
    if sum(x['files_min'] for x in groups) != inventory['total_files_minimum']:
        raise CorpusGateError('root/child file totals mismatch')
    if sum(x['bytes_min'] for x in groups) != inventory['total_size_bytes_minimum']:
        raise CorpusGateError('root/child byte totals mismatch')
    return {
        'schema': SCHEMA,
        'manifest_kind': 'FOLDER_LEVEL_PREPARATION',
        'source_inventory_date': inventory['date'],
        'user_training_authorization': 'GRANTED_2026-10-26',
        'third_party_rights': 'NOT_INDEPENDENTLY_VERIFIED',
        'public_distribution_permitted': False,
        'cloud_gpu_spend_authorized': False,
        'training_started_from_drive': False,
        'full_file_manifest_complete': False,
        'roots': [
            {'code': code, 'drive_folder_id': approved_roots[code],
             'files_min': sources[code]['files_min'],
             'bytes_min': sources[code]['size_bytes_min'],
             'folder_listing_complete': sources[code]['complete']}
            for code in approved_roots
        ],
        'group_count': len(groups),
        'total_files_min': inventory['total_files_minimum'],
        'total_bytes_min': inventory['total_size_bytes_minimum'],
        'formats_min': inventory['formats_minimum'],
        'groups': groups,
        'holds': [
            'DRIVE_LISTING_TRUNCATED',
            'NO_SHA256_FOR_DRIVE_MEDIA',
            'NO_CONTENT_BASED_CROSS_ARCHIVE_DEDUP',
            'NO_SOURCE_RIGHTS_AUDIT',
            'NO_CAMERA_COLOR_CALIBRATION',
            'NO_APPROVED_FINAL_DATASET_SPLIT'
        ]
    }


def paginated_folder_files(service: Any, folder_id: str, *,
                           max_pages: int = 10000) -> list[dict]:
    """Read all Drive v3 list pages, detecting duplicate IDs and token loops."""
    if not isinstance(folder_id, str) or not re.fullmatch(
            r'[A-Za-z0-9_-]{12,160}', folder_id):
        raise CorpusGateError('invalid folder identifier')
    result = []
    seen_ids = set()
    seen_tokens = set()
    token = None
    for _ in range(max_pages):
        request = {
            'q': f"'{folder_id}' in parents and trashed = false",
            'pageSize': 1000,
            'supportsAllDrives': True,
            'includeItemsFromAllDrives': True,
            'fields': 'nextPageToken,files(id,name,mimeType,size,md5Checksum,parents,modifiedTime)'
        }
        if token is not None:
            request['pageToken'] = token
        response = service.files().list(**request).execute()
        if not isinstance(response, dict) or not isinstance(response.get('files'), list):
            raise CorpusGateError('malformed Drive listing response')
        for item in response['files']:
            fid = item.get('id')
            if not fid or fid in seen_ids:
                raise CorpusGateError('duplicate/missing Drive item ID')
            seen_ids.add(fid)
            result.append(item)
        next_token = response.get('nextPageToken')
        if not next_token:
            return result
        if not isinstance(next_token, str) or next_token in seen_tokens:
            raise CorpusGateError('invalid/repeated Drive nextPageToken')
        seen_tokens.add(next_token)
        token = next_token
    raise CorpusGateError('pagination incomplete; maximum pages exceeded')


def enumerate_drive(service: Any, roots: dict[str, str]) -> list[dict]:
    """Walk approved roots; do not download any media payload."""
    result = []
    seen_folders = set()
    seen_files = set()
    for root_code, root_folder in roots.items():
        pending = [root_folder]
        while pending:
            folder = pending.pop()
            if folder in seen_folders:
                raise CorpusGateError('repeated or cyclic source folder')
            seen_folders.add(folder)
            for item in paginated_folder_files(service, folder):
                if item.get('mimeType') == 'application/vnd.google-apps.folder':
                    pending.append(item['id'])
                    continue
                identifier = item['id']
                if identifier in seen_files:
                    raise CorpusGateError('duplicate file across roots')
                seen_files.add(identifier)
                suffix = str(item.get('name') or '').rsplit('.', 1)[-1].upper()
                if suffix not in _FORMATS:
                    continue
                size = int(item.get('size') or 0)
                if size <= 0:
                    raise CorpusGateError('missing media size')
                md5 = item.get('md5Checksum')
                if md5 is not None and not _MD5_RE.fullmatch(md5):
                    raise CorpusGateError('invalid Drive MD5')
                result.append({
                    'drive_id': identifier,
                    'root': root_code,
                    'group_id': folder,
                    'filename': item['name'],
                    'media_class': _FORMATS[suffix],
                    'format': suffix,
                    'size_bytes': size,
                    'drive_md5_untrusted': md5,
                    'content_sha256': None,
                    'split_hint': _split_hint(folder),
                    'final_split': None,
                    'consent_from_uploader': True,
                    'training_allowed': False,
                    'holds': [
                        'SHA256_REQUIRED',
                        'SOURCE_LICENSE_AND_SUBJECT_REVIEW',
                        'CROSS_ARCHIVE_DUPLICATES_NOT_RESOLVED'
                    ]
                })
    result.sort(key=lambda x: (x['root'], x['group_id'], x['drive_id']))
    return result


def validate_file_catalog(entries: list[dict]) -> dict:
    """Report possible duplicates by size+MD5 without certifying uniqueness."""
    seen_ids = set()
    group_splits = defaultdict(set)
    digests = defaultdict(list)
    for item in entries:
        if item['drive_id'] in seen_ids:
            raise CorpusGateError('duplicated Drive file ID')
        seen_ids.add(item['drive_id'])
        group_splits[item['group_id']].add(item['split_hint'])
        if item['drive_md5_untrusted']:
            digests[(item['size_bytes'],
                     item['drive_md5_untrusted'].lower())].append(item['drive_id'])
    if any(len(x) != 1 for x in group_splits.values()):
        raise CorpusGateError('split-hint leakage within folder')
    return {
        'schema': SCHEMA,
        'record_count': len(entries),
        'counts_by_format': dict(Counter(x['format'] for x in entries)),
        'possible_duplicate_sets_by_drive_md5':
            [ids for ids in digests.values() if len(ids) > 1],
        'training_allowed': False,
        'note': 'MD5 is only a hint, NOT verified content SHA256.'
    }


def write_private_json_exclusive(path: str | Path, document: dict) -> Path:
    """Create once at mode 0600; refuse overwrite and symlinks."""
    destination = Path(path)
    if destination.is_symlink() or destination.exists():
        raise CorpusGateError('manifest overwrite refused')
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + '\n'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    fd = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description='AETHERIS private training manifest, read-only Drive metadata')
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--consent', required=True, type=Path)
    parser.add_argument('--roots', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scan-drive', action='store_true')
    parser.add_argument('--files-output', type=Path)
    args = parser.parse_args(argv)
    approved_roots = json.loads(args.roots.read_text(encoding='utf-8'))
    inventory = json.loads(args.inventory.read_text(encoding='utf-8'))
    consent = json.loads(args.consent.read_text(encoding='utf-8'))
    plan = plan_from_inventory(inventory, consent=consent,
                               approved_roots=approved_roots)
    if args.scan_drive:
        if args.files_output is None:
            parser.error('--scan-drive requires --files-output')
        import google.auth
        from googleapiclient.discovery import build
        credentials, _ = google.auth.default(
            scopes=['https://www.googleapis.com/auth/drive.metadata.readonly'])
        service = build('drive', 'v3', credentials=credentials,
                        cache_discovery=False)
        records = enumerate_drive(service, approved_roots)
        stats = validate_file_catalog(records)
        formats = Counter(x['format'] for x in records)
        if (len(records) < inventory['total_files_minimum'] or
                any(formats[key] < count
                    for key, count in inventory['formats_minimum'].items())):
            raise CorpusGateError('live enumeration smaller than inventory lower bound')
        file_manifest = {
            'schema': SCHEMA, 'manifest_kind': 'PER_FILE_DRIVE_METADATA',
            'training_allowed': False, 'records': records, 'validation': stats,
            'source_inventory_lower_bound': inventory['total_files_minimum']
        }
        write_private_json_exclusive(args.files_output, file_manifest)
        plan['full_file_manifest_complete'] = True
        plan['full_file_manifest_count'] = len(records)
        plan['manifest_kind'] = 'FOLDER_PREPARATION_PLUS_COMPLETE_FILE_METADATA'
    elif args.files_output is not None:
        parser.error('--files-output requires --scan-drive')
    write_private_json_exclusive(args.output, plan)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
