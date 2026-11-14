"""Measure this machine's CPU serving memory. No paid API calls; no model changes."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import sys,json,time,threading,resource,platform,base64
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import psutil
PROCESS=psutil.Process();peak=0;done=threading.Event();phase='imports';samples=[];phases=[]
def sample():
    global peak
    while not done.wait(.05):
        rss=PROCESS.memory_info().rss;peak=max(peak,rss);samples.append((phase,rss))
thread=threading.Thread(target=sample,daemon=True);thread.start()
def run(name,fn):
    global phase
    phase=name;start=time.perf_counter();answer=fn()
    current=[rss for p,rss in samples if p==name]
    phases.append({'phase':name,'seconds':round(time.perf_counter()-start,3),'peak_rss_mib':round(max(current or [PROCESS.memory_info().rss])/1024**2,1),'end_rss_mib':round(PROCESS.memory_info().rss/1024**2,1)})
    print(phases[-1],flush=True);return answer
import torch
torch.set_num_threads(2)
from api.model_service import ModelService
from api.multimodal import MultimodalService
from api.speech_main import transcribe_bytes,decode_bounded
from scripts.validate_multimodal import frame_samples
from io import BytesIO
import numpy as np
import av
service=ModelService(('bow_nb','tfidf_lr','improved_lr','bilstm','bert'))
bonus=MultimodalService()
run('load_five_text_models',service.load)
clip=ROOT/'data/bonus_validation/AudioWAV/1001_DFA_HAP_XX.wav'
audio=decode_bounded(clip.read_bytes())
def wav(seconds):
    a=np.tile(audio,int(np.ceil(seconds*16000/len(audio))))[:seconds*16000]
    out=BytesIO()
    with av.open(out,'w',format='wav') as c:
        stream=c.add_stream('pcm_s16le',rate=16000)
        frame=av.AudioFrame.from_ndarray((np.clip(a,-1,1)*32767).astype(np.int16).reshape(1,-1),format='s16',layout='mono');frame.sample_rate=16000
        for packet in stream.encode(frame):c.mux(packet)
        for packet in stream.encode(None):c.mux(packet)
    return out.getvalue()
frames=frame_samples(ROOT/'data/bonus_validation/VideoFlash/1001_DFA_HAP_XX.flv')
audio20=base64.b64encode(wav(20)).decode();audio60=wav(60)
text=('I feel happy about my presentation but I am worried about tomorrow. '*40)[:2000]
run('text_2000_characters',lambda:service.predict(text,list(service.model_ids)))
run('expression_three_frames_cold',lambda:bonus.face(frames))
run('acoustic_20_seconds_cold',lambda:bonus.acoustic(audio20))
run('whisper_60_seconds_cold',lambda:transcribe_bytes(audio60))
def together(a,b):
    with ThreadPoolExecutor(2)as pool:
        futures=[pool.submit(a),pool.submit(b)]
        return [f.result()for f in futures]
run('two_requests_text_and_acoustic',lambda:together(lambda:service.predict(text,list(service.model_ids)),lambda:bonus.acoustic(audio20)))
run('two_requests_whisper_and_acoustic',lambda:together(lambda:transcribe_bytes(audio60),lambda:bonus.acoustic(audio20)))
run('two_requests_whisper_and_expression',lambda:together(lambda:transcribe_bytes(audio60),lambda:bonus.face(frames)))
done.set();thread.join()
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
report={'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'cpu_threads':2,'peak_sampled_rss_mib':round(peak/1024**2,1),'os_high_water_rss_mib':round(rss/(1024**2 if sys.platform=='darwin'else 1024),1),'phases':phases,'limitations':['macOS ARM measurement, not a Linux container memory guarantee.','Two simultaneous inferences match the hosted admission limit; repeat on chosen Render instance.','No DeepSeek calls made; no training or calibration changes.','Repeated public acted audio exercises input limits, not recognition accuracy.']}
(ROOT/'docs/reproducibility/render_memory_profile.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
