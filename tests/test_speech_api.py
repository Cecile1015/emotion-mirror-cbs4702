"""Speech input limits; no model download, microphone or paid calls."""
from io import BytesIO
import wave
import numpy as np
from fastapi.testclient import TestClient
from api import speech_main as speech


def wav(seconds=1):
    data=BytesIO()
    with wave.open(data,'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000)
        stream.writeframes(np.zeros(int(seconds*16000),dtype=np.int16).tobytes())
    return data.getvalue()


def test_decode_duration_and_silence(monkeypatch,tmp_path):
    (tmp_path/'model.bin').touch();monkeypatch.setattr(speech,'MODEL_DIR',tmp_path)
    assert speech.decode_bounded(wav()).shape == (16000,)
    with TestClient(speech.app) as client:
        assert client.post('/transcribe',content=wav(63),headers={'Content-Type':'audio/wav'}).status_code==422
        assert client.post('/transcribe',content=wav(),headers={'Content-Type':'audio/wav'}).status_code==422


def test_media_validation_and_size_limits(monkeypatch,tmp_path):
    (tmp_path/'model.bin').touch();monkeypatch.setattr(speech,'MODEL_DIR',tmp_path)
    with TestClient(speech.app) as client:
        assert client.post('/transcribe',content=b'x',headers={'Content-Type':'text/plain'}).status_code==415
        assert client.post('/transcribe',content=b'',headers={'Content-Type':'audio/webm'}).status_code==422
        assert client.post('/transcribe',content=b'not audio',headers={'Content-Type':'audio/webm'}).status_code==422
        assert client.post('/transcribe',content=b'x'*(speech.MAX_BYTES+1),headers={'Content-Type':'audio/webm'}).status_code==413


def test_missing_model_is_explicit(monkeypatch,tmp_path):
    monkeypatch.setattr(speech,'MODEL_DIR',tmp_path)
    with TestClient(speech.app) as client:
        assert client.get('/health').json()['configured'] is False
        assert client.post('/transcribe',content=wav(),headers={'Content-Type':'audio/wav'}).status_code==503
