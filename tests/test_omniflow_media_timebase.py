"""Rational timeline regression for video/audio in native OmniFlow."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"omniflow-worker"))
from omega_sdk.media_timebase import (
    RationalTimebase, rescale_timestamp, TimebaseError,
    validate_decode_timestamps, validate_presentation_timestamps, FrameStamp,
)


class MediaTimebaseTests(unittest.TestCase):
    def test_exact_23976_video_frame_to_audio_samples(self):
        self.assertEqual(rescale_timestamp(1,RationalTimebase(1001,24000),
                                           RationalTimebase(1,48000)),2002)
        self.assertEqual(rescale_timestamp(24000,RationalTimebase(1001,24000),
                                           RationalTimebase(1,48000)),48048000)

    def test_reduce_rational_timebases_and_reject_invalid(self):
        self.assertEqual(RationalTimebase(2,100).num,1)
        self.assertEqual(RationalTimebase(2,100).den,50)
        for n,d in ((0,1),(1,0),(1,-4),(-2,4),(True,5),(1,False),(2**65,1)):
            with self.subTest(n=n,d=d),self.assertRaises(TimebaseError):
                RationalTimebase(n,d)

    def test_b_frame_pts_reordering_is_not_rejected_in_decode_order(self):
        stamp=(FrameStamp(0,0),FrameStamp(3,1),FrameStamp(1,2),FrameStamp(2,3))
        validate_decode_timestamps(stamp)
        with self.assertRaises(TimebaseError):
            validate_presentation_timestamps(stamp)
        validate_presentation_timestamps(sorted(stamp,key=lambda x:x.pts))

    def test_duplicate_dts_rejected(self):
        with self.assertRaises(TimebaseError):
            validate_decode_timestamps((FrameStamp(2,3),FrameStamp(4,3)))

    def test_nearest_even_without_float_drift(self):
        src=RationalTimebase(1,2);dst=RationalTimebase(1,1)
        self.assertEqual(rescale_timestamp(1,src,dst,'nearest_even'),0)
        self.assertEqual(rescale_timestamp(3,src,dst,'nearest_even'),2)
        self.assertEqual(rescale_timestamp(-3,src,dst,'nearest_even'),-2)

    def test_floor_ceil_and_negative_pts(self):
        src=RationalTimebase(1,3);dst=RationalTimebase(1,1)
        self.assertEqual(rescale_timestamp(-1,src,dst,'floor'),-1)
        self.assertEqual(rescale_timestamp(-1,src,dst,'ceil'),0)
        self.assertEqual(rescale_timestamp(2,src,dst,'floor'),0)
        self.assertEqual(rescale_timestamp(2,src,dst,'ceil'),1)

    def test_int64_overflow_fails_closed(self):
        src=RationalTimebase(1,1);dst=RationalTimebase(1,2)
        with self.assertRaises(TimebaseError):
            rescale_timestamp(2**63-1,src,dst)
        with self.assertRaises(TimebaseError):
            rescale_timestamp(True,src,dst)

    def test_unknown_rounding_rejected(self):
        with self.assertRaises(TimebaseError):
            rescale_timestamp(1,RationalTimebase(1,1),RationalTimebase(1,1),'approximate')
