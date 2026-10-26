"""Native AETHERIS temporal contract: rational audio/video timestamps, no float drift.

Supports presentation-order validation and decode-order (B-frame) DTS
validation separately; timestamp arithmetic checks signed int64 bounds.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import gcd
from typing import Iterable

I64_MIN=-(1<<63)
I64_MAX=(1<<63)-1

class TimebaseError(ValueError):
    pass


def _i64(value:int,label:str)->int:
    if type(value) is not int or not I64_MIN<=value<=I64_MAX:
        raise TimebaseError(f'{label} must fit signed 64-bit integer')
    return value


@dataclass(frozen=True)
class RationalTimebase:
    """Positive number of seconds per tick, normalized by GCD."""
    num:int
    den:int
    def __post_init__(self):
        num=_i64(self.num,'timebase numerator')
        den=_i64(self.den,'timebase denominator')
        if num<=0 or den<=0:
            raise TimebaseError('timebase numerator and denominator must be positive')
        common=gcd(num,den)
        object.__setattr__(self,'num',num//common)
        object.__setattr__(self,'den',den//common)


@dataclass(frozen=True)
class FrameStamp:
    pts:int
    dts:int
    def __post_init__(self):
        _i64(self.pts,'presentation timestamp')
        _i64(self.dts,'decode timestamp')


def rescale_timestamp(value:int,source:RationalTimebase,target:RationalTimebase,
                      rounding:str='nearest_even')->int:
    """Rescale integer ticks using arbitrary precision, return exact i64.

    roundings: nearest_even, floor, ceil. Negative timestamps are valid
    (e.g., preroll). No binary float is used in timestamp conversion.
    """
    val=_i64(value,'timestamp')
    if not isinstance(source,RationalTimebase) or not isinstance(target,RationalTimebase):
        raise TimebaseError('source and target timebases required')
    if rounding not in ('nearest_even','floor','ceil'):
        raise TimebaseError('unknown rounding policy')
    n=val*source.num*target.den
    d=source.den*target.num
    if rounding=='floor':
        result=n//d
    elif rounding=='ceil':
        result=-(-n//d)
    else:
        sign=-1 if n<0 else 1
        magnitude=abs(n)
        integral,remainder=divmod(magnitude,d)
        if 2*remainder>d or (2*remainder==d and integral%2):
            integral+=1
        result=sign*integral
    return _i64(result,'rescaled timestamp')


def validate_decode_timestamps(frames:Iterable[FrameStamp])->None:
    """DTS must strictly advance in decode order; PTS may be reordered."""
    previous=None
    for frame in frames:
        if not isinstance(frame,FrameStamp):
            raise TimebaseError('FrameStamp required')
        if previous is not None and frame.dts<=previous:
            raise TimebaseError('DTS must strictly increase in decode order')
        previous=frame.dts


def validate_presentation_timestamps(frames:Iterable[FrameStamp])->None:
    """PTS must advance if caller presents frames in display order."""
    previous=None
    for frame in frames:
        if not isinstance(frame,FrameStamp):
            raise TimebaseError('FrameStamp required')
        if previous is not None and frame.pts<=previous:
            raise TimebaseError('PTS must strictly increase in display order')
        previous=frame.pts
