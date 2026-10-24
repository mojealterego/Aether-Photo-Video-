"""S24 Ultra acceptance runner: measured RSS, disk, thermal and TIFF integrity.

Linux/Termux POSIX only. Missing thermal sensors are recorded as null; an
Android LMK killing the Python process cannot be reported as a passed test.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
from typing import Any
from dataclasses import dataclass
import tifffile

def _read_kib(path:Path,key:str)->int|None:
    try:
        for line in path.read_text().splitlines():
            if line.startswith(key+':'):
                return int(line.split()[1])
    except (OSError,ValueError,IndexError):
        pass
    return None

def thermal_celsius()->float|None:
    values=[]
    for sensor in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            t=float(sensor.read_text().strip())/1000.
            if -20<t<140: values.append(t)
        except (OSError,ValueError): pass
    return max(values) if values else None

def free_bytes(path:Path)->int:
    p=path.resolve()
    while not p.exists() and p!=p.parent: p=p.parent
    stat=os.statvfs(p)
    return stat.f_bavail*stat.f_frsize

def read_device_stats()->dict[str,Any]:
    rss=_read_kib(Path('/proc/self/status'),'VmRSS')
    available=_read_kib(Path('/proc/meminfo'),'MemAvailable')
    peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) if sys.platform.startswith('linux') else None
    return dict(rss_kib=rss,peak_rss_os_kib=peak,system_available_ram_kib=available,
                thermal_max_c=thermal_celsius())

@dataclass(frozen=True)
class AcceptanceConfig:
    source:Path
    output:Path
    workdir:Path
    report:Path
    target_mp:float=400.
    width:int|None=None
    height:int|None=None
    tile:int=256
    reserve_free_gb:float=1.
    sample_interval:float=.25

    def __post_init__(self):
        for key in ('source','output','workdir','report'):
            object.__setattr__(self,key,Path(getattr(self,key)).expanduser())
        if not .05<=self.sample_interval<=10:
            raise ValueError('Sampling interval must be between .05 and 10s')
        if len({p.resolve() for p in (self.source,self.output,self.report)})!=3:
            raise ValueError('Input, output and report paths must differ')
        if self.report.exists():raise FileExistsError(self.report)

def measured_render(config:AcceptanceConfig,*,render=None)->dict[str,Any]:
    from mobile_s24.s24_render import MobileConfig,run
    if not config.source.is_file():raise FileNotFoundError(config.source)
    if config.output.exists():raise FileExistsError(config.output)
    config.report.parent.mkdir(parents=True,exist_ok=True)
    free_before=free_bytes(config.workdir)
    peak_rss=read_device_stats()['rss_kib'] or 0
    peak_temp=thermal_celsius()
    samples=0
    stop=threading.Event()
    def sample()->None:
        nonlocal peak_rss,peak_temp,samples
        while not stop.is_set():
            stats=read_device_stats()
            peak_rss=max(peak_rss,stats['rss_kib'] or 0)
            if stats['thermal_max_c'] is not None:
                peak_temp=max([x for x in (peak_temp,stats['thermal_max_c']) if x is not None])
            samples+=1
            stop.wait(config.sample_interval)
    cfg=MobileConfig(config.source,config.output,config.workdir,
                     target_width=config.width,target_height=config.height,
                     target_mp=config.target_mp,tile_size=config.tile,
                     reserve_free_gb=config.reserve_free_gb)
    thread=threading.Thread(target=sample,daemon=True)
    started=time.monotonic()
    thread.start()
    try:
        details=(render or run)(cfg)
        with tifffile.TiffFile(config.output) as f:
            p=f.pages[0]
            if not f.is_bigtiff or p.dtype.name!='uint16' or len(p.shape)!=3 or p.shape[2]!=3:
                raise RuntimeError('BigTIFF RGB16 verification failed')
            dimensions=list(p.shape)
    finally:
        stop.set()
        thread.join(timeout=2.)
    final=read_device_stats()
    peak_rss=max(peak_rss,final['rss_kib'] or 0)
    report={'status':'passed_local_execution','platform':sys.platform,
            'device_model':'not_authenticated','output_shape_hwc':dimensions,
            'seconds':round(time.monotonic()-started,3),
            'rss_sample_peak_kib':peak_rss,'rss_os_peak_kib':final['peak_rss_os_kib'],
            'thermal_max_c':peak_temp,'thermal_sensor_available':peak_temp is not None,
            'samples':samples,'free_before_bytes':free_before,
            'free_after_bytes':free_bytes(config.workdir),
            'render':details,'no_oom_guarantee':True}
    temp=config.report.with_name(config.report.name+'.partial')
    with temp.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,config.report)
    return report

def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description='Offline S24 400MP RAM/thermal/disk acceptance')
    for name in ('source','output','workdir','report'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--target-mp',type=float,default=400.)
    p.add_argument('--width',type=int)
    p.add_argument('--height',type=int)
    p.add_argument('--tile',type=int,default=256)
    p.add_argument('--reserve-free-gb',type=float,default=1.)
    a=p.parse_args(argv)
    rep=measured_render(AcceptanceConfig(a.source,a.output,a.workdir,a.report,
                 target_mp=a.target_mp,width=a.width,height=a.height,tile=a.tile,
                 reserve_free_gb=a.reserve_free_gb))
    print(json.dumps(rep,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
