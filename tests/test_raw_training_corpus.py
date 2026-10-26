"""AETHERIS RAW training grant/split contract — synthetic metadata only."""
import unittest
from omega_sdk.vae.raw_training_corpus import (
    CorpusSource,TrainingGrant,build_split_plan,CorpusAdmissionError
)

ROOT='synthetic-test-root-0001'
GRANT=TrainingGrant(ROOT,True,'PHOTO_VAE_TRAINING','USER_EXPLICIT_CHAT_CONSENT')
def source(n,folder=None,digest=None,ext='.NEF'):
    return CorpusSource(file_id=f'file{n}',folder_id=folder or f'folder{n%6}',
                        filename=f'image{n}{ext}',byte_size=12000000,sha256=digest)

class Plans(unittest.TestCase):
    def test_explicit_authorization_required(self):
        with self.assertRaises(CorpusAdmissionError):
            build_split_plan([source(i) for i in range(12)],
                TrainingGrant(ROOT,False,'PHOTO_VAE_TRAINING','UNKNOWN'))
    def test_train_val_test_grouped_by_parent_folder(self):
        entries=[source(i,folder=f'folder{i//3}') for i in range(93)]
        plan=build_split_plan(entries,GRANT)
        lookup={e['file_id']:e['split'] for e in plan['files']}
        self.assertEqual(len(lookup),len(entries))
        for g in range(31):
            self.assertEqual(len({lookup[f'file{i}'] for i in range(3*g,3*g+3)}),1)
        self.assertTrue({'train','validation','test'}<=set(lookup.values()))
    def test_nef_only_jpg_and_video_denied(self):
        for ext in ['.JPG','.MP4','.DNG']:
            with self.subTest(ext=ext),self.assertRaises(CorpusAdmissionError):
                build_split_plan([source(i,ext=ext) for i in range(10)],GRANT)
    def test_duplicate_ids_and_duplicate_sha_rejected(self):
        entries=[source(i,folder=f'folder{i}',digest='a'*64) for i in range(11)]
        with self.assertRaises(CorpusAdmissionError):build_split_plan(entries,GRANT)
        entries=[source(i,folder=f'folder{i}') for i in range(11)]
        entries.append(source(3))
        with self.assertRaises(CorpusAdmissionError):build_split_plan(entries,GRANT)
    def test_metadata_only_not_certified_for_training(self):
        plan=build_split_plan([source(i,folder=f'folder{i}') for i in range(12)],GRANT)
        self.assertEqual(plan['status'],'PROVISIONAL_METADATA_ONLY')
        self.assertFalse(plan['ready_for_production_training'])
        self.assertEqual(plan['missing_file_sha256'],12)
    def test_rejects_wrong_root(self):
        with self.assertRaises(CorpusAdmissionError):
            build_split_plan([source(i,folder=f'folder{i}') for i in range(10)],
                             GRANT,root_folder_id='foreign')
    def test_plan_stable_in_any_listing_order(self):
        entries=[source(i,folder=f'folder{i}') for i in range(31)]
        self.assertEqual(build_split_plan(entries,GRANT),
                         build_split_plan(list(reversed(entries)),GRANT))
    def test_training_grant_does_not_extend_to_tc16(self):
        with self.assertRaises(CorpusAdmissionError):
            build_split_plan([source(i) for i in range(10)],
                TrainingGrant(ROOT,True,'VIDEO_TC16_TRAINING','USER_EXPLICIT_CHAT_CONSENT'))

if __name__=='__main__':unittest.main()
