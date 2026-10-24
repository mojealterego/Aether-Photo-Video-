"""Video per-frame offline Aether BigTIFF restoration and SHA256 recovery.

Maintains a single RGB16 frame in mobile renderer, uses same spatial/optical
pipeline as photographs. Temporal neural smoothing is not yet implemented.
"""
from __future__ import annotations
import json,os,re
from pathlib import Path
import tifffile
from .frame_pipeline import _sha

_PATTERN=re.compile(r'^frame_\d{6}\.tif$')

def _atomic(path:Path,state:dict):
    temp=path.with_name(path.name+'.partial')
    with temp.open('x',encoding='utf8') as stream:
        json.dump(state,stream,sort_keys=True,indent=2)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)

def _render_frame(source:Path,destination:Path,wh:tuple[int,int],tile:int):
    from mobile_s24.s24_render import MobileConfig,run
    width,height=wh
    job=MobileConfig(source,destination,destination.parent/'render_work'/destination.stem,
        target_width=width,target_height=height,tile_size=tile,
        reserve_free_gb=0)
    run(job)

def process_video_frames(frames_dir:str|Path,output_dir:str|Path,*,
                         width:int,height:int,tile:int=256,render_frame=None)->dict:
    source,destination=Path(frames_dir),Path(output_dir)
    if not source.is_dir() or source.resolve()==destination.resolve():
        raise ValueError('Source and destination video frame directories must differ')
    if not 2<=width<=8192 or not 2<=height<=8192 or width*height>16_000_000:
        raise ValueError('Frame geometry above mobile processing cap')
    if tile%16 or not 16<=tile<=1024:raise ValueError('Tile must be 16..1024 multiple of 16')
    manifest=json.loads((source/'manifest.json').read_text())
    if (manifest.get('version')!=1 or not isinstance(manifest.get('frames'),list)
        or not manifest['frames'] or not isinstance(manifest.get('source_sha256'),str)
        or len(manifest['source_sha256'])!=64 or not isinstance(manifest.get('info'),dict)):
        raise ValueError('Invalid source frame manifest')
    for i,entry in enumerate(manifest['frames']):
        name=f'frame_{i:06d}.tif'
        if entry.get('name')!=name or not _PATTERN.fullmatch(name):
            raise ValueError('Noncanonical source frame name')
        file=source/name
        if not file.exists() or _sha(file)!=entry.get('sha256'):
            raise ValueError('Missing or changed source frame')
    signature={'source_manifest_sha256':_sha(source/'manifest.json'),
               'width':width,'height':height,'tile':tile}
    destination.mkdir(parents=True,exist_ok=True)
    journal=destination/'progress.json'
    if journal.exists():
        state=json.loads(journal.read_text())
        if state.get('signature')!=signature:
            raise ValueError('Frame render signature changed')
    else:
        if any(destination.iterdir()):raise ValueError('Untracked output files')
        state={'signature':signature,'done':[]}
        _atomic(journal,state)
    if not isinstance(state.get('done'),list) or len(state['done'])>len(manifest['frames']):
        raise ValueError('Corrupt frame journal')
    reused=len(state['done'])
    for i,entry in enumerate(state['done']):
        name=f'frame_{i:06d}.tif'
        output=destination/name
        if entry.get('name')!=name or not output.exists() or _sha(output)!=entry.get('sha256'):
            raise ValueError('Checkpoint output has changed')
    for i in range(reused,len(manifest['frames'])):
        name=f'frame_{i:06d}.tif'
        final=destination/name
        temp=destination/(name+'.partial.tif')
        if final.exists() or temp.exists():
            raise FileExistsError('Untracked or incomplete frame output')
        (render_frame or _render_frame)(source/name,temp,(width,height),tile)
        with tifffile.TiffFile(temp) as tif:
            p=tif.pages[0]
            if p.shape!=(height,width,3) or p.dtype.name!='uint16':
                raise ValueError('Rendered output must be RGB16')
        os.replace(temp,final)
        state['done'].append({'name':name,'sha256':_sha(final)})
        _atomic(journal,state)
    new={**manifest,'frames':list(state['done']),
         'original_dimensions':[manifest['info']['width'],manifest['info']['height']],
         'output_dimensions':[width,height],'processing':'aether_framewise_rgb16'}
    new['info']={**manifest['info'],'width':width,'height':height}
    target=destination/'manifest.json'
    if target.exists():
        if json.loads(target.read_text())!=new:
            raise ValueError('Output manifest incompatible with resume')
    else:_atomic(target,new)
    return {'frames':len(new['frames']),'reused':reused,
            'executed':len(new['frames'])-reused,'output_directory':str(destination),
            'format':'RGB16 BigTIFF'}
