"""Read-only TC16 footage intake: exact timing, hashes and admission gates.

Uploader media is VALIDATION_ONLY by default. No original files are modified,
no learning/training runs, and the report should remain private unless the
copyright holder explicitly authorizes publication.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from typing import Iterable


class MediaIntakeError(ValueError):
    """Malformed/unsupported media or failed read-only verification."""


_ALLOWED_SUFFIXES = frozenset({'.mp4', '.mov', '.mkv'})
_MAX_FILE_BYTES = 20 * (1 << 30)
_MAX_METADATA_BYTES = 2 * (1 << 20)


def tc16_time_contract(frames: int) -> dict:
    """TC16 anchor + full 4-frame groups; remaining EOS frames need repair.

    Native TC16 currently uses last-frame-hold for the remainder; therefore
    these tail frames are NOT reconstructed from explicit source latents.
    """
    if type(frames) is not int or frames < 1 or frames > 10_000_000:
        raise MediaIntakeError('invalid frame count')
    groups = (frames - 1) // 4
    decoded = 1 + 4 * groups
    return {
        'anchor_frames': 1,
        'temporal_stride': 4,
        'latent_frames': 1 + groups,
        'decoded_from_groups': decoded,
        'tail_frames_not_encoded': frames - decoded,
        'tail_strategy_current': 'LAST_FRAME_HOLD_LOSSY',
        'bit_exact_tail_restoration': False,
    }


def _run(args: list[str], *, timeout: int) -> bytes:
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaIntakeError('media verification unavailable or timed out') from exc
    if result.returncode != 0:
        raise MediaIntakeError('ffprobe/ffmpeg could not decode media')
    return result.stdout


def audit_video(path: Path, *, verify_decode: bool = False) -> dict:
    source = Path(path)
    if source.suffix.lower() not in _ALLOWED_SUFFIXES or source.is_symlink():
        raise MediaIntakeError('unsupported extension or symlink input')
    try:
        details = source.stat()
    except OSError as exc:
        raise MediaIntakeError('media file not found') from exc
    if not stat.S_ISREG(details.st_mode) or not 0 < details.st_size <= _MAX_FILE_BYTES:
        raise MediaIntakeError('media file size/type outside bounded admission')
    raw = _run(['ffprobe','-v','error','-show_streams',
                '-show_format','-of','json',str(source)],timeout=45)
    if len(raw) > _MAX_METADATA_BYTES:
        raise MediaIntakeError('oversized container metadata')
    try:
        meta = json.loads(raw)
        streams = meta['streams']
        videos = [item for item in streams if item.get('codec_type') == 'video'
                  and item.get('disposition', {}).get('attached_pic', 0) != 1]
        if len(videos) != 1:
            raise MediaIntakeError('exactly one primary video stream required')
        video = videos[0]
        width = int(video['width']); height = int(video['height'])
        if not 1 <= width <= 32768 or not 1 <= height <= 32768:
            raise MediaIntakeError('invalid dimensions')
        frames_raw = video.get('nb_frames')
        if frames_raw is None:
            raise MediaIntakeError('unknown frame count; explicit counted probe required')
        frames = int(frames_raw)
        if str(frames) != str(frames_raw):
            raise MediaIntakeError('noncanonical frame count')
        tc16 = tc16_time_contract(frames)
        fps = Fraction(video['avg_frame_rate'])
        if fps <= 0 or fps > 1000 or fps.denominator > 1_000_000:
            raise MediaIntakeError('invalid FPS timebase')
        duration = float(meta['format']['duration'])
        if not 0 < duration <= 3600 * 12:
            raise MediaIntakeError('unbounded or invalid duration')
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        if isinstance(exc, MediaIntakeError):
            raise
        raise MediaIntakeError('invalid source timing/geometry') from exc
    hash_acc = hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            hash_acc.update(chunk)
    result = {
        'filename': source.name,
        'sha256': hash_acc.hexdigest(),
        'size_bytes': details.st_size,
        'codec': video.get('codec_name', 'unknown'),
        'profile': video.get('profile'),
        'width': width, 'height': height,
        'fps_fraction': f'{fps.numerator}/{fps.denominator}',
        'frames': frames,
        'duration_seconds': duration,
        'audio_tracks': [{k: s.get(k) for k in ('codec_name','sample_rate','channels')}
                         for s in streams if s.get('codec_type') == 'audio'],
        'color_primaries': video.get('color_primaries', 'unspecified'),
        'color_transfer': video.get('color_transfer', 'unspecified'),
        'color_matrix': video.get('color_space', 'unspecified'),
        'color_range': video.get('color_range', 'unspecified'),
        'tc16': tc16,
        'permitted_use': 'VALIDATION_ONLY',
        'training_approved': False,
        'decode_check': 'NOT_RUN',
    }
    if verify_decode:
        _run(['ffmpeg','-nostdin','-hide_banner','-v','error','-xerror',
              '-i',str(source),'-map','0:v:0','-map','0:a?',
              '-f','null','-'],timeout=180)
        result['decode_check'] = 'PASS'
    return result


def audit_files(paths: Iterable[Path], *, verify_decode: bool = False) -> dict:
    sources = list(paths)
    if not sources or len(sources) > 1000:
        raise MediaIntakeError('invalid intake batch size')
    names = set()
    digests = set()
    records = []
    for path in sources:
        resolved = str(Path(path).absolute())
        if resolved in names:
            raise MediaIntakeError('duplicate input path')
        names.add(resolved)
        record = audit_video(Path(path), verify_decode=verify_decode)
        if record['sha256'] in digests:
            raise MediaIntakeError('byte-identical media duplicates in intake')
        digests.add(record['sha256'])
        records.append(record)
    return {
        'schema': 'aether.tc16.video_intake/1.0',
        'training_permission': 'NOT_GRANTED',
        'originals_immutable': True,
        'files': records,
        'summary': {
            'clips': len(records),
            'frames': sum(r['frames'] for r in records),
            'duration_seconds': round(sum(r['duration_seconds'] for r in records), 6),
            'tc16_latent_frames': sum(r['tc16']['latent_frames'] for r in records),
            'tc16_tail_frames_not_encoded': sum(r['tc16']['tail_frames_not_encoded'] for r in records),
            'clips_with_audio': sum(bool(r['audio_tracks']) for r in records),
            'clips_with_unspecified_color_primaries': sum(r['color_primaries'] == 'unspecified' for r in records),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Read-only intake CLI, optionally publishing one new manifest exclusively."""
    import argparse
    parser = argparse.ArgumentParser(description='AETHERIS TC16 read-only media intake')
    parser.add_argument('sources', type=Path, nargs='+')
    parser.add_argument('--verify-decode', action='store_true')
    parser.add_argument('--output', type=Path, help='new JSON manifest path (never overwrite)')
    args = parser.parse_args(argv)
    report = audit_files(args.sources, verify_decode=args.verify_decode)
    wire = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output is None:
        print(wire, end='')
    else:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(wire)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
