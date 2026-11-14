import base64
from io import BytesIO
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from api.multimodal import MultimodalRequest, MultimodalService, acoustic_features, decode64, image_from_bytes, Frame
from api.speech_main import app
from api.public_guard import RemoteBudget, PublicGuard


def test_channel_consent_and_bounds():
    with pytest.raises(ValidationError):MultimodalRequest(capture_id='a',channels=['face'],audio='abcd')
    with pytest.raises(ValidationError):MultimodalRequest(capture_id='a',channels=['face','face'])
    with pytest.raises(ValidationError):Frame(offset_ms=22000,image='abcd')
    with pytest.raises(ValueError):decode64('not-base64',100)
    with pytest.raises(ValueError):decode64(base64.b64encode(b'1234').decode(),3)


def test_descriptive_pitch_not_emotion():
    tone=.1*np.sin(2*np.pi*200*np.arange(16000)/16000)
    features=acoustic_features(tone)
    assert abs(features['median_f0_hz']-200)<3
    assert 'emotion'not in features
    assert acoustic_features(np.zeros(16000))['median_f0_hz']is None


def test_unsafe_or_corrupt_image_rejected():
    with pytest.raises(ValueError):image_from_bytes(b'corrupt')
    b=BytesIO();Image.new('RGB',(1601,48)).save(b,format='PNG')
    with pytest.raises(ValueError):image_from_bytes(b.getvalue())


def test_partial_failure_retains_success(monkeypatch):
    service=MultimodalService()
    monkeypatch.setattr(service,'health',lambda:{'channels':{'face':{'configured':True},'acoustic':{'configured':True}}})
    monkeypatch.setattr(service,'face',lambda frames:(_ for _ in ()).throw(ValueError('No clear frontal face')))
    monkeypatch.setattr(service,'acoustic',lambda audio:{'channel':'acoustic','label':'neu'})
    r=service.predict(MultimodalRequest(capture_id='test',channels=['face','acoustic']))
    assert r['results'][0]['label']=='neu' and r['errors'][0]['channel']=='face' and r['fusion']is None


def test_model_busy_rejected():
    service=MultimodalService();service.lock.acquire()
    try:
        with pytest.raises(Exception)as e:service.predict(MultimodalRequest(capture_id='x',channels=['face']))
        assert e.value.status_code==429
    finally:service.lock.release()


def test_no_face_and_multiple_faces(monkeypatch):
    image=BytesIO();Image.new('RGB',(200,100)).save(image,format='JPEG')
    frame=Frame(offset_ms=0,image=base64.b64encode(image.getvalue()).decode())
    service=MultimodalService()
    for boxes,message in [([], 'No clear'), ([(2,2,50,50),(100,2,50,50)], 'possible face region')]:
        monkeypatch.setattr(service,'face_boxes',lambda img:boxes)
        with pytest.raises(ValueError,match=message):service.face([frame])


def test_yunet_rejects_weak_detections(monkeypatch):
    class Detector:
        def setInputSize(self,size):self.size=size
        def detect(self,pixels):
            rows=np.zeros((3,15)); rows[:,:4]=[[10,10,80,80],[150,10,80,80],[250,10,10,10]]
            rows[:,-1]=[.95,.5,.99]
            return None,rows
    service=MultimodalService();service.detector=Detector()
    assert len(service.face_boxes(Image.new('RGB',(640,480))))==1


def test_silent_audio_abstains():
    import av
    output=BytesIO()
    with av.open(output,'w',format='wav')as container:
        stream=container.add_stream('pcm_s16le',rate=16000)
        frame=av.AudioFrame.from_ndarray(np.zeros((1,32000),dtype=np.int16),format='s16',layout='mono');frame.sample_rate=16000
        for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode(None):container.mux(packet)
    with pytest.raises(ValueError,match='silent'):MultimodalService().acoustic(base64.b64encode(output.getvalue()).decode())


def test_media_api_validation():
    client=TestClient(app)
    assert client.get('/multimodal/health').json()['max_seconds']==20
    assert client.post('/multimodal',json={'capture_id':'x','channels':['acoustic'],'frames':[{'offset_ms':0,'image':'abcd'}]}).status_code==422
    assert client.post('/multimodal',content=b'{}').status_code==415
    r=client.post('/multimodal',json={'capture_id':'x','channels':['face']})
    assert r.status_code==200 and r.json()['errors'] and not r.json()['results']


def test_remote_budget_includes_repair(monkeypatch):
    monkeypatch.setenv('EMOTION_DEEPSEEK_SESSION_CALL_LIMIT','2');budget=RemoteBudget();budget.reserve()
    with pytest.raises(RuntimeError):budget.reserve()


def test_public_size_and_origin_guards():
    from fastapi import FastAPI
    api=FastAPI();api.add_middleware(PublicGuard)
    @api.post('/api/predict')
    def predict():return {'ok':True}
    client=TestClient(api)
    assert client.post('/api/predict',content=b'x'*16001).status_code==413
    assert client.post('/api/predict',headers={'origin':'https://example.com'}).status_code==403
    for _ in range(20):assert client.post('/api/predict',content='{}').status_code==200
    assert client.post('/api/predict',content='{}').status_code==429


def test_ambiguous_frame_does_not_discard_clear_frame(monkeypatch):
    import torch
    from types import SimpleNamespace
    image=BytesIO();Image.new('RGB',(200,100)).save(image,format='JPEG')
    payload=base64.b64encode(image.getvalue()).decode()
    frames=[Frame(offset_ms=i*1000,image=payload) for i in range(2)]
    service=MultimodalService()
    boxes=iter([[(2,2,50,50),(100,2,50,50)],[(40,10,60,60)]])
    monkeypatch.setattr(service,'face_boxes',lambda img:next(boxes))
    service.face_processor=lambda **kwargs:{}
    service.face_model=lambda **kwargs:SimpleNamespace(logits=torch.tensor([[0.,0.,0.,3.,0.,0.,0.]]))
    result=service.face(frames)
    assert result['frames_used']==1 and result['frame_offsets_ms']==[1000]
    assert result['frames_ambiguous']==[0] and any('excluded'in w for w in result['warnings'])
