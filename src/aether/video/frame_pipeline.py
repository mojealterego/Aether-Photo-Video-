"""Frame-wise video backend for S24 Ultra: FFmpeg, RGB48LE, TIFF16, FFV1.

Constant-frame-rate videos only. No complete video held in process memory.
VFR, Dolby Vision/HDR metadata transfer and temporal AI are not yet supported.
"""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import hashlib,json,os,subprocess
import numpy as np
import tifffile

class VideoUnsupported(ValueError):
    pass

def _command(cmd:list[str],cap:int=4*1024*1024)->bytes:
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if r.returncode:raise RuntimeError(r.stderr[-1200:].decode('utf-8','replace'))
    if len(r.stdout)>cap:raise VideoUnsupported('FFmpeg metadata too large')
    return r.stdout

def _sha(path:Path)->str:
    dig=hashlib.sha256()
    with path.open('rb') as handle:
        for blob in iter(lambda:handle.read(1024*1024),b''):dig.update(blob)
    return dig.hexdigest()

def inspect_video(source:str|Path)->dict:
    source=Path(source)
    if not source.is_file():raise FileNotFoundError(source)
    raw=_command(['ffprobe','-v','error','-show_entries',
                  'stream=codec_type,width,height,avg_frame_rate,r_frame_rate,codec_name:format=duration',
                  '-of','json',str(source)])
    doc=json.loads(raw)
    video=[x for x in doc.get('streams',[]) if x.get('codec_type')=='video']
    if len(video)!=1:raise VideoUnsupported('Exactly one video stream required')
    v=video[0]
    w,h=int(v['width']),int(v['height'])
    rate=Fraction(v['avg_frame_rate'])
    if min(w,h)<2 or max(w,h)>16384 or not 0<rate<=240:
        raise VideoUnsupported('Invalid geometry/frame-rate')
    pts_raw=_command(['ffprobe','-v','error','-select_streams','v:0',
               '-read_intervals','%+#40','-show_entries','frame=best_effort_timestamp_time',
               '-of','csv=p=0',str(source)],cap=32768)
    pts=[float(v.split(',')[0]) for v in pts_raw.decode().splitlines()
         if v.strip() and v.split(',')[0]!='N/A']
    if len(pts)>=3 and np.any(np.abs(np.diff(pts)-float(1/rate))>.002):
        raise VideoUnsupported('VFR video requires timestamp-aware exporter')
    return {'width':w,'height':h,'fps':str(rate),'audio':any(x.get('codec_type')=='audio'
            for x in doc.get('streams',[])),'video_codec':v.get('codec_name')}

def _atomic_json(path:Path,doc:dict):
    temp=path.with_name(path.name+'.partial')
    with temp.open('x',encoding='utf-8') as h:
        json.dump(doc,h,sort_keys=True,indent=2)
        h.flush();os.fsync(h.fileno())
    os.replace(temp,path)

def extract_frames_rgb16(source:str|Path,out_dir:str|Path,*,
                         max_frames:int=3600,max_pixels:int=8_847_360)->dict:
    source,root=Path(source),Path(out_dir)
    info=inspect_video(source)
    w,h=info['width'],info['height']
    if not 1<=max_frames<=500000 or w*h>max_pixels:
        raise VideoUnsupported('Video resource budget exceeded')
    if root.exists():
        manifest=root/'manifest.json'
        if not manifest.is_file():raise ValueError('Untracked frame directory')
        old=json.loads(manifest.read_text())
        if old['source_sha256']!=_sha(source) or old['max_frames']!=max_frames:
            raise ValueError('Changed video source/settings during resume')
        for record in old['frames']:
            path=root/record['name']
            if not path.is_file() or _sha(path)!=record['sha256']:
                raise ValueError('Missing/corrupt extracted frame')
        return {'frames':len(old['frames']),'reused':True,'fps':info['fps']}
    root.mkdir(parents=True)
    command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i',str(source),
             '-map','0:v:0','-an','-sn','-dn','-vsync','0',
             '-pix_fmt','rgb48le','-f','rawvideo','pipe:1']
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    count=w*h*3*2
    records=[]
    try:
        for i in range(max_frames+1):
            data=bytearray()
            while len(data)<count:
                chunk=proc.stdout.read(count-len(data))
                if not chunk:break
                data.extend(chunk)
            if not data:break
            if len(data)!=count:raise ValueError('Truncated video frame')
            if i==max_frames:raise VideoUnsupported('Maximum frame count exceeded')
            array=np.frombuffer(data,dtype='<u2').reshape(h,w,3)
            path=root/f'frame_{i:06d}.tif'
            tifffile.imwrite(path,array,photometric='rgb',metadata=None,bigtiff=True)
            records.append({'name':path.name,'sha256':_sha(path)})
        code=proc.wait(timeout=60)
        if code or not records:raise RuntimeError('FFmpeg decode did not complete')
        _atomic_json(root/'manifest.json',{'version':1,
                    'source_sha256':_sha(source),'max_frames':max_frames,'info':info,'frames':records})
        return {'frames':len(records),'fps':info['fps'],'reused':False}
    except BaseException:
        if proc.poll() is None:proc.kill();proc.wait()
        raise
    finally:
        if proc.stdout:proc.stdout.close()

def mux_frames_lossless(frames_dir:str|Path,source:str|Path,output:str|Path)->dict:
    root,source,output=map(Path,(frames_dir,source,output))
    if output.exists():raise FileExistsError(output)
    if output.suffix.lower()!='.mkv':raise VideoUnsupported('Lossless output requires .mkv')
    ledger=json.loads((root/'manifest.json').read_text())
    if ledger['source_sha256']!=_sha(source):raise ValueError('Original video changed')
    for rec in ledger['frames']:
        if _sha(root/rec['name'])!=rec['sha256']:
            raise ValueError('Corrupt frame ledger')
    w,h=ledger['info']['width'],ledger['info']['height']
    fps=ledger['info']['fps']
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.stem+'.partial.mkv')
    if tmp.exists():raise FileExistsError(tmp)
    args=['ffmpeg','-hide_banner','-loglevel','error','-nostdin',
          '-f','rawvideo','-pixel_format','rgb48le','-video_size',f'{w}x{h}',
          '-framerate',fps,'-i','pipe:0','-i',str(source),
          '-map','0:v:0','-map','1:a?','-c:v','ffv1','-level','3',
          '-c:a','copy','-shortest',str(tmp)]
    proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL)
    try:
        for record in ledger['frames']:
            image=tifffile.imread(root/record['name'])
            if image.shape!=(h,w,3) or image.dtype!=np.dtype('uint16'):
                raise VideoUnsupported('Incorrect frame geometry or type')
            proc.stdin.write(np.asarray(image,dtype='<u2').tobytes())
        proc.stdin.close()
        if proc.wait(timeout=120):raise RuntimeError('Lossless video encoder failed')
        checked=inspect_video(tmp)
        if (checked['width'],checked['height'])!=(w,h):
            raise RuntimeError('Encoded video dimensions changed')
        os.replace(tmp,output)
        return {'output':str(output),'video_codec':'ffv1','frames':len(ledger['frames']),
                'fps':fps,'audio_copied':checked['audio']}
    except BaseException:
        if proc.poll() is None:proc.kill();proc.wait()
        raise
    finally:
        if tmp.exists():tmp.unlink()
