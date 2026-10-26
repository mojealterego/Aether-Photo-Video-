"""AETHERIS Drive corpus preparation: no customer data, fake Drive pages."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from omega_sdk.vae.drive_corpus import (
    CorpusGateError, enumerate_drive, paginated_folder_files,
    plan_from_inventory, validate_file_catalog, write_private_json_exclusive
)


class FakeDrive:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.calls = []
    def files(self):
        return self
    def list(self, **kwargs):
        self.calls.append(kwargs)
        return self
    def execute(self):
        return next(self.pages)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.roots = {
            'A': 'root_abcdefghij_A', 'B': 'root_abcdefghij_B',
            'C': 'root_abcdefghij_C', 'D': 'root_abcdefghij_D',
            'E': 'root_abcdefghij_E', 'F': 'root_abcdefghij_F'
        }
        self.inventory = {
            'schema': 'aether.raw_drive_metadata_inventory/1.0',
            'date': '2026-10-26',
            'roots': [
                {'root': key,
                 'url': 'https://drive.google.com/drive/folders/' + value,
                 'files_min': 1, 'size_bytes_min': 100,
                 'complete': key != 'D'}
                for key, value in self.roots.items()
            ],
            'subfolders': [
                {'root': key, 'subfolder_code': '01',
                 'counts': {'NEF': 1}, 'files_min': 1,
                 'size_bytes_min': 100, 'complete': key != 'D'}
                for key in self.roots
            ],
            'formats_minimum': {'NEF': 6},
            'total_files_minimum': 6,
            'total_size_bytes_minimum': 600
        }
        self.consent = {'authorization_from_uploader': 'GRANTED'}

    def build(self):
        return plan_from_inventory(
            self.inventory, consent=self.consent,
            approved_roots=self.roots)

    def test_plan_keeps_training_blocked_and_groups_disjoint(self):
        result = self.build()
        self.assertEqual(result['total_files_min'], 6)
        self.assertEqual(result['group_count'], 6)
        self.assertFalse(result['training_started_from_drive'])
        self.assertFalse(result['full_file_manifest_complete'])
        self.assertTrue(all(not g['trainable'] for g in result['groups']))
        self.assertTrue(all(g['final_split'] is None for g in result['groups']))

    def test_explicit_consent_is_mandatory(self):
        self.consent['authorization_from_uploader'] = 'DENIED'
        with self.assertRaises(CorpusGateError):
            self.build()

    def test_wrong_root_or_byte_total_fails_closed(self):
        broken = deepcopy(self.inventory)
        broken['roots'][0]['url'] += '_MISMATCH'
        with self.assertRaises(CorpusGateError):
            plan_from_inventory(broken, consent=self.consent,
                                approved_roots=self.roots)
        broken = deepcopy(self.inventory)
        broken['total_size_bytes_minimum'] += 1
        with self.assertRaises(CorpusGateError):
            plan_from_inventory(broken, consent=self.consent,
                                approved_roots=self.roots)

    def test_split_hints_are_repeatable_not_approved(self):
        x, y = self.build(), self.build()
        self.assertEqual([g['split_hint'] for g in x['groups']],
                         [g['split_hint'] for g in y['groups']])
        self.assertTrue(any('DRIVE_PAGINATION_REQUIRED' in g['hold_reasons']
                            for g in x['groups']))

    def test_two_page_listing_reads_next_page_token(self):
        fake = FakeDrive([
            {'files': [{'id': 'first'}], 'nextPageToken': 'next'},
            {'files': [{'id': 'second'}]}
        ])
        got = paginated_folder_files(fake, 'folder_abcdefghi')
        self.assertEqual([x['id'] for x in got], ['first', 'second'])
        self.assertEqual(fake.calls[1]['pageToken'], 'next')

    def test_repeated_ids_fail_even_when_pages_are_distinct(self):
        fake = FakeDrive([
            {'files': [{'id': 'a'}], 'nextPageToken': 'next'},
            {'files': [{'id': 'a'}]}
        ])
        with self.assertRaises(CorpusGateError):
            paginated_folder_files(fake, 'folder_abcdefghi')

    def test_reused_next_page_token_fails(self):
        fake = FakeDrive([
            {'files': [], 'nextPageToken': 'next'},
            {'files': [], 'nextPageToken': 'next'}
        ])
        with self.assertRaises(CorpusGateError):
            paginated_folder_files(fake, 'folder_abcdefghi')

    def test_page_limit_fails_instead_of_approving_partial_data(self):
        fake = FakeDrive([{'files': [], 'nextPageToken': 'next'}])
        with self.assertRaises(CorpusGateError):
            paginated_folder_files(fake, 'folder_abcdefghi', max_pages=1)

    def test_enumerated_file_metadata_remains_non_trainable(self):
        root = self.roots['A']
        fake = FakeDrive([{'files': [
            {'id': 'raw1', 'name': 'image.NEF', 'size': '100',
             'mimeType': 'application/octet-stream'},
            {'id': 'video2', 'name': 'movie.mp4', 'size': '100',
             'mimeType': 'video/mp4'}
        ]}])
        items = enumerate_drive(fake, {'A': root})
        self.assertEqual(len(items), 2)
        self.assertTrue(all(x['training_allowed'] is False for x in items))
        self.assertTrue(all(x['content_sha256'] is None for x in items))
        self.assertEqual(items[0]['split_hint'], items[1]['split_hint'])

    def test_metadata_md5_dedup_is_a_warning_not_sha256_proof(self):
        entries = [
            {'drive_id': 'file1', 'group_id': 'g1', 'split_hint': 'train',
             'drive_md5_untrusted': 'f' * 32, 'size_bytes': 200,
             'format': 'NEF'},
            {'drive_id': 'file2', 'group_id': 'g2', 'split_hint': 'val',
             'drive_md5_untrusted': 'f' * 32, 'size_bytes': 200,
             'format': 'NEF'}
        ]
        result = validate_file_catalog(entries)
        self.assertFalse(result['training_allowed'])
        self.assertEqual(result['possible_duplicate_sets_by_drive_md5'],
                         [['file1', 'file2']])

    def test_private_manifest_is_exclusive_and_mode_0600(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'private.json'
            write_private_json_exclusive(target, self.build())
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(target.read_text())['group_count'], 6)
            with self.assertRaises(CorpusGateError):
                write_private_json_exclusive(target, {})


if __name__ == '__main__':
    unittest.main()
