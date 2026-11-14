"""One public origin for the exported website and all CPU inference routes."""
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('EMOTION_DEEPSEEK_SESSION_CALL_LIMIT', '100')
import torch
torch.set_num_threads(int(os.getenv('EMOTION_CPU_THREADS', '2')))
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from .main import create_app
from .model_service import ModelService
from .speech_main import app as media_app, health as speech_health
from .multimodal import service as bonus_service
from .public_guard import PublicGuard
ROOT=Path(__file__).resolve().parents[1]

def create_hosted_app(service=None, static_dir=None):
    service=service if service is not None else ModelService()
    text_app=create_app(service)
    @asynccontextmanager
    async def lifespan(app):
        async with text_app.router.lifespan_context(text_app):
            yield
    app=FastAPI(title='Emotion Mirror',lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    app.add_middleware(PublicGuard)
    for route in text_app.routes:
        if getattr(route,'path','')in ('/health','/models','/predict'):
            app.add_api_route('/api'+route.path,route.endpoint,methods=list(route.methods),response_model=route.response_model)
    for route in media_app.routes:
        if getattr(route,'path','')in ('/transcribe','/multimodal','/multimodal/health','/health'):
            suffix='/speech-health'if route.path=='/health'else route.path
            app.add_api_route('/api'+suffix,route.endpoint,methods=list(route.methods))
    @app.get('/healthz')
    def readiness(response:Response):
        ready=service.health()['ready'] and speech_health()['configured'] and all(x['configured']for x in bonus_service.health()['channels'].values())
        response.headers['Cache-Control']='no-store'
        if not ready:response.status_code=503
        # No remote credit-consuming health probe. First media inference is lazy.
        return {'ready':ready,'media_models':'files present; loaded on first use','remote_connectivity_checked':False}
    @app.get('/api/evidence')
    def evidence():
        return {name:json.loads((ROOT/'web/data'/file).read_text())for name,file in [('ledger','ledger.json'),('errors','error_analysis.json')]}
    static_dir=Path(static_dir or os.getenv('EMOTION_STATIC_DIR',str(ROOT/'web/out')))
    if not (static_dir/'index.html').exists():raise RuntimeError('Build the static website before starting the hosted service.')
    app.mount('/',StaticFiles(directory=static_dir,html=True),name='website')
    return app
