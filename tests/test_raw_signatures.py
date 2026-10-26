"""AETHERIS raw signature tests; synthetic headers only, no private RAW."""
import struct
import tempfile
import unittest
from pathlib import Path
from aether.ingest.raw_signature import (
    classify_raw_container, RawContainerKind, RawSignatureError
)

class RawSignatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name: str, data: bytes) -> Path:
        path = self.root / name
        path.write_bytes(data)
        return path

    def test_native_cr2_not_inferred_from_extension_alone(self):
        cr2 = b'II*\x00' + struct.pack('<I',16) + b'CR\x02\x00' + (b'\x00'*24)
        info = classify_raw_container(self.write('camera.CR2',cr2))
        self.assertEqual(info.kind, RawContainerKind.CANON_CR2)
        self.assertTrue(info.standard_container)

    def test_native_ciff_crw(self):
        p=self.write('camera.CRW',b'II\x1a\x00\x00\x00HEAPCCDR'+b'\x00'*30)
        self.assertEqual(classify_raw_container(p).kind,RawContainerKind.CANON_CIFF_CRW)

    def test_dng_requires_version_tag_not_only_tiff_header(self):
        ifd=struct.pack('<H',1)+struct.pack('<HHI',50706,1,4)+b'\x01\x04\x00\x00'+struct.pack('<I',0)
        info=classify_raw_container(self.write('camera.DNG',b'II*\x00'+struct.pack('<I',8)+ifd))
        self.assertEqual(info.kind,RawContainerKind.DNG)
        self.assertTrue(info.standard_container)

    def test_no_dng_tag_is_unclassified_tiff(self):
        ifd=struct.pack('<H',1)+struct.pack('<HHI',256,4,1)+struct.pack('<I',100)+struct.pack('<I',0)
        p=self.write('not_dng.DNG',b'II*\x00'+struct.pack('<I',8)+ifd)
        self.assertEqual(classify_raw_container(p).kind,RawContainerKind.UNCLASSIFIED_TIFF)

    def test_headerless_cr2_detected_but_not_auto_trusted(self):
        info=classify_raw_container(self.write('chdk.CR2',b'\x87\xaa'*64))
        self.assertEqual(info.kind,RawContainerKind.HEADERLESS_SENSOR_DUMP)
        self.assertFalse(info.admitted_by_signature)

    def test_headerless_crw_detected_but_not_auto_trusted(self):
        info=classify_raw_container(self.write('chdk.CRW',b'\x82\x07\x7c\x00'*30))
        self.assertEqual(info.kind,RawContainerKind.HEADERLESS_SENSOR_DUMP)
        self.assertFalse(info.admitted_by_signature)

    def test_fake_suffix_no_extension_based_admission(self):
        self.assertFalse(classify_raw_container(self.write('payload.CR2',b'not a raw image')).admitted_by_signature)

    def test_broken_tiff_bounded_parser_fails_closed(self):
        p=self.write('broken.DNG',b'II*\x00'+struct.pack('<I',2**31)+b'bad')
        self.assertEqual(classify_raw_container(p).kind,RawContainerKind.UNCLASSIFIED_TIFF)

    def test_empty_and_nonexistent_files_fail(self):
        for p in [self.write('empty.CR2',b''),self.root/'missing.CR2']:
            with self.subTest(path=p),self.assertRaises(RawSignatureError):
                classify_raw_container(p)

    def test_extension_disagreement_flagged(self):
        p=self.write('mislabeled.CRW',b'II*\x00'+struct.pack('<I',16)+b'CR\x02\x00'+b'\x00'*24)
        info=classify_raw_container(p)
        self.assertEqual(info.kind,RawContainerKind.CANON_CR2)
        self.assertTrue(info.extension_mismatch)
        self.assertFalse(info.admitted_by_signature)

if __name__=='__main__':
    unittest.main()
