"""Exercise the deployed contract with public CREMA-D fixtures; no paid LLM call."""
import base64,json,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.validate_multimodal import frame_samples
base=sys.argv[1]if len(sys.argv)>1 else 'http://127.0.0.1:7860'
clip='1001_DFA_HAP_XX';audio=(ROOT/f'data/bonus_validation/AudioWAV/{clip}.wav').read_bytes()
def send(path,body,mime):
 req=urllib.request.Request(base+path,data=body,headers={'Content-Type':mime})
 with urllib.request.urlopen(req,timeout=120)as r:return json.load(r)
transcript=send('/api/transcribe',audio,'audio/wav')
text=send('/api/predict',json.dumps({'text':transcript['text'],'models':['tfidf_lr','bert']}).encode(),'application/json')
media=send('/api/multimodal',json.dumps({'capture_id':'cremad-public-fixture','channels':['face','acoustic'],'frames':[f.model_dump()for f in frame_samples(ROOT/f'data/bonus_validation/VideoFlash/{clip}.flv')],'audio':base64.b64encode(audio).decode()}).encode(),'application/json')
assert len(text['results'])==2 and not text['errors'],text
assert len(media['results'])==2 and not media['errors'],media
report={'base':base,'fixture':clip,'transcript':transcript,'text':text,'multimodal':media,'note':'Public acted fixture used for API integration, not a live browser microphone/camera test.'}
name='space_public_smoke.json'if base.startswith('https:')else'space_local_smoke.json'
(ROOT/'docs/reproducibility'/name).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'base':base,'transcript':transcript['text'],'text':[(r['model_id'],r['label'])for r in text['results']],'bonus':[(r['channel'],r['label'])for r in media['results']]},indent=2))
