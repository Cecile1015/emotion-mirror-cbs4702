"""Fixed external smoke benchmark, not training or model selection on test data.

CREMA-D: actors 1001..1006, sentence DFA, all six acted labels, XX intensity.
Face: 36 clips / six represented classes. Acoustic: 24 / four supported classes.
No score-based example selection or tuning. Raw media are ignored by Git.
"""
from pathlib import Path
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import av
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from api.multimodal import Frame, MultimodalService

REV='1658cd342dff90010aa843eaeebd53610a08b1dc'
SOURCE='https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/'+REV
DEST=ROOT/'data/bonus_validation'
FACE={'ANG':'angry','DIS':'disgust','FEA':'fear','HAP':'happy','NEU':'neutral','SAD':'sad'}
AUDIO={'ANG':'ang','HAP':'hap','NEU':'neu','SAD':'sad'}

def fetch(relative):
    path=DEST/relative;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(SOURCE+'/'+relative,timeout=90) as r:path.write_bytes(r.read())
    if path.read_bytes().startswith(b'version https://git-lfs'):raise RuntimeError('LFS pointer, not media')
    return {'path':relative,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def frame_samples(path):
    with av.open(str(path)) as c:
        # Small public fixture only; API uses bounded still-image decoding.
        frames=list(c.decode(video=0))
    out=[]
    for i in sorted(set([len(frames)//4,len(frames)//2,3*len(frames)//4])):
        image=frames[i].to_image();buf=BytesIO();image.save(buf,format='JPEG',quality=80)
        out.append(Frame(offset_ms=round(float(frames[i].time or 0)*1000),image=base64.b64encode(buf.getvalue()).decode()))
    return out

def metrics(rows,channel,labels):
    selected=[r for r in rows if r['channel']==channel];valid=[r for r in selected if 'prediction'in r]
    gold=[r['gold']for r in valid];pred=[r['prediction']for r in valid]
    return {'attempted':len(selected),'predicted':len(valid),'coverage':len(valid)/len(selected),
            'accuracy_on_predicted':accuracy_score(gold,pred) if valid else None,
            'accuracy_counting_failures_incorrect':sum(g==p for g,p in zip(gold,pred))/len(selected),
            'macro_f1_on_predicted_represented_classes':f1_score(gold,pred,labels=labels,average='macro',zero_division=0) if valid else None,
            'labels':labels,'confusion_labels':labels+(['surprise'] if channel=='face' else []),'confusion_matrix':confusion_matrix(gold,pred,labels=labels+(['surprise'] if channel=='face' else [])).tolist() if valid else [],
            'warning':'Small fixed convenience sample of acted clips; labels are intended expressions, not inner feelings. Not a deployment accuracy estimate.'}

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'docs/reproducibility/multimodal_validation.json');args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    names=[f'{actor}_DFA_{emotion}_XX'for actor in range(1001,1007)for emotion in FACE]
    paths=[f'VideoFlash/{n}.flv'for n in names]+[f'AudioWAV/{n}.wav'for n in names if n.split('_')[2]in AUDIO]
    with ThreadPoolExecutor(max_workers=4)as pool: files=list(pool.map(fetch,paths))
    print('Downloaded',len(files),'fixtures',flush=True)
    torch.set_num_threads(2);service=MultimodalService();rows=[]
    for n in names:
        emotion=n.split('_')[2]
        for channel in ['face']+(['acoustic']if emotion in AUDIO else []):
            row={'clip':n,'channel':channel,'gold':(FACE if channel=='face'else AUDIO)[emotion]}
            try:
                r=service.face(frame_samples(DEST/f'VideoFlash/{n}.flv'))if channel=='face'else service.acoustic(base64.b64encode((DEST/f'AudioWAV/{n}.wav').read_bytes()).decode())
                row.update(prediction=r['label'],confidence=r['confidence'])
                if channel=='face':row['frames_used']=r['frames_used']
            except Exception as e:row['error']=str(e)
            rows.append(row);print(n,channel,row.get('prediction',row.get('error')),flush=True)
    report={'detector':'OpenCV YuNet 2023mar; confidence 0.9, NMS 0.3; excludes ambiguous frames', 'evaluation_note':'Regression re-evaluation of the existing fixed sample after a detector repair; not an independent held-out model-selection test.', 'dataset':'CREMA-D','source':'https://github.com/CheyneyComputerScience/CREMA-D','revision':REV,'license':'ODbL-1.0 / DbCL-1.0',
            'protocol':__doc__,'selection':'Predeclared actors 1001–1006, DFA, XX; no tuning. Intended/acted filename labels, not crowd-perceived majority labels.',
            'face':metrics(rows,'face',list(FACE.values())), 'acoustic':metrics(rows,'acoustic',list(AUDIO.values())),
            'limitations':['No surprise reference examples in CREMA-D; surprise predictions are still possible.','Six actors, one sentence, acted English speech; not representative of users, spontaneous speech or all demographic groups.','Face model trained on FER2013; acoustic model on IEMOCAP. Camera crop and chunking differ from their original benchmarks.','Abstentions/errors counted separately; F1 only uses successfully predicted examples.'],
            'predictions':rows,'files':files}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in ('face','acoustic')},indent=2))
if __name__=='__main__':main()
