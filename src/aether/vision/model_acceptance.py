"""Evidence-gated ONNX model smoke tests for S24 or optional remote worker.

Checks genuine model forward execution only, NOT identity preservation,
face recovery, depth accuracy, or CodeFormer perceptual quality. No weights
are bundled. A CodeFormer ONNX input/output adapter export is required.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
_INPUT={'depth':(1,3,518,518),'arcface':(1,3,112,112),'codeformer':(1,3,512,512)}

def _sha(path):
    dig=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for b in iter(lambda:handle.read(1048576),b''):dig.update(b)
    return dig.hexdigest()

def validate_session(session,kind:str)->dict:
    if kind not in _INPUT:raise ValueError('Unsupported model kind')
    inputs=session.get_inputs()
    if len(inputs)!=1 or not session.get_outputs():
        raise ValueError('ONNX must expose one RGB input and an output')
    shape=inputs[0].shape
    if len(shape)!=4 or (isinstance(shape[1],int) and shape[1]!=3):
        raise ValueError('ONNX input must be NCHW RGB')
    required=_INPUT[kind]
    if kind=='depth':
        wh=[shape[i] if isinstance(shape[i],int) and shape[i]>0 else 518 for i in (2,3)]
        if min(wh)<32 or max(wh)>1024:raise ValueError('Depth ONNX geometry outside mobile budget')
        required=(1,3,*wh)
    else:
        for n,v in zip(shape,required):
            if isinstance(n,int) and n!=v:raise ValueError('Wrong fixed model geometry')
    if np.prod(required)>4_000_000:raise ValueError('Large model input rejected')
    started=time.perf_counter()
    result=session.run(None,{inputs[0].name:np.zeros(required,dtype='f4')})
    latency=round(1000*(time.perf_counter()-started),2)
    if not result:raise ValueError('Inference returned nothing')
    y=np.asarray(result[0],dtype='f4')
    if not np.isfinite(y).all():raise ValueError('Invalid model result')
    if kind=='depth':
        z=y.squeeze()
        if z.ndim!=2 or min(z.shape)<2:raise ValueError('Bad depth output')
    elif kind=='arcface':
        if y.size!=512 or float(np.linalg.norm(y))<1e-8:
            raise ValueError('Bad ArcFace feature vector')
    elif y.shape!=(1,3,512,512):
        raise ValueError('CodeFormer ONNX export requires RGB512 NCHW output')
    return {'kind':kind,'status':'executed','input_shape':list(required),
            'output_shape':list(y.shape),'latency_ms':latency,
            'validation_scope':'real_forward_pass_only'}

def check_evidence(evidence:dict,*,required:list[str])->bool:
    for key in required:
        record=evidence.get(key)
        if (not isinstance(record,dict) or record.get('status')!='executed'
            or not isinstance(record.get('model_sha256'),str) or len(record['model_sha256'])!=64
            or not record.get('provider')):
            raise ValueError('Missing real model execution evidence: '+key)
    return True

def run_checkpoint(path:Path,kind:str,provider:str='CPUExecutionProvider')->dict:
    path=Path(path)
    if not path.is_file() or path.suffix.lower()!='.onnx':
        raise FileNotFoundError('Explicit compatible ONNX checkpoint required')
    try:import onnxruntime as ort
    except ImportError as ex:raise RuntimeError('Install ONNX Runtime on execution device') from ex
    if provider not in ort.get_available_providers():
        raise RuntimeError('ONNX execution provider unavailable: '+provider)
    session=ort.InferenceSession(str(path),providers=[provider])
    report=validate_session(session,kind)
    report.update({'model_sha256':_sha(path),'provider':provider,'model_path':str(path)})
    return report

def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description='Actual ONNX model execution acceptance (not perceptual validation)')
    for name in _INPUT:p.add_argument('--'+name,type=Path)
    p.add_argument('--provider',default='CPUExecutionProvider')
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args(argv)
    selected={key:getattr(a,key) for key in _INPUT if getattr(a,key) is not None}
    if not selected:p.error('At least one model checkpoint must be specified')
    if a.report.exists():raise FileExistsError(a.report)
    report={name:run_checkpoint(path,name,a.provider) for name,path in selected.items()}
    check_evidence(report,required=list(selected))
    a.report.parent.mkdir(parents=True,exist_ok=True)
    partial=a.report.with_name(a.report.name+'.partial')
    with partial.open('x',encoding='utf-8') as f:
        json.dump(report,f,sort_keys=True,indent=2)
        f.flush();os.fsync(f.fileno())
    os.replace(partial,a.report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
