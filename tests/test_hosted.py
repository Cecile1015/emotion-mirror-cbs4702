from pathlib import Path
from fastapi.testclient import TestClient
from api.hosted import create_hosted_app

class Service:
    def __init__(self):self.started=False
    def load(self):self.started=True
    def close(self):self.started=False
    def health(self):return {'ready':self.started}
    def registry(self):return {'models':[]}
    def predict(self,*args):raise AssertionError('No paid/model calls in route tests')

def test_hosted_routes_and_static_isolation(tmp_path,monkeypatch):
    (tmp_path/'index.html').write_text('<h1>Emotion Mirror</h1>')
    import api.hosted as module
    monkeypatch.setattr(module,'speech_health',lambda:{'configured':True})
    monkeypatch.setattr(module.bonus_service,'health',lambda:{'channels':{'face':{'configured':True},'acoustic':{'configured':True}}})
    service=Service()
    with TestClient(create_hosted_app(service,tmp_path))as client:
        assert client.get('/').status_code==200
        assert client.get('/api/models').json()=={'models':[]}
        assert client.get('/healthz').json()['ready']is True
        assert client.get('/.env').status_code==404
        assert client.get('/models/bert/model.safetensors').status_code==404
        assert client.get('/docs').status_code==404
        assert client.post('/api/predict',content='x'*16001).status_code==413
    assert not service.started

def test_readiness_rejects_missing_media(tmp_path,monkeypatch):
    import api.hosted as module
    (tmp_path/'index.html').write_text('test')
    monkeypatch.setattr(module,'speech_health',lambda:{'configured':False})
    with TestClient(create_hosted_app(Service(),tmp_path))as client:
        assert client.get('/healthz').status_code==503

def test_download_rejects_hash_mismatch(tmp_path,monkeypatch):
    import json,hashlib
    from io import BytesIO
    from deployment.render import fetch_models
    source=tmp_path/'manifest.json';source.write_text(json.dumps([{'path':'models/a.bin','bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest(),'url':'https://example.invalid/a'}]))
    monkeypatch.setattr(fetch_models,'MANIFEST',source)
    monkeypatch.setattr(fetch_models.urllib.request,'urlopen',lambda *a,**k:BytesIO(b'bad'))
    import pytest
    with pytest.raises(RuntimeError,match='hash mismatch'):fetch_models.prepare(tmp_path)
    assert not (tmp_path/'models/a.bin').exists()
    assert not (tmp_path/'models/a.bin.download').exists()
