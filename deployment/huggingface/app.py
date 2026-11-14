"""Hybrid Gradio Space: the BooDarling-inspired static React UI is the landing page.

An actual Gradio service-inspector is mounted at /service; all prediction routes
use the same FastAPI service. CPU inference avoids a CUDA requirement. The Space
must be eligible for its configured hardware/plan; this code does not buy compute.
"""
import json
import os
from pathlib import Path
from contextlib import asynccontextmanager
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('EMOTION_DEEPSEEK_SESSION_CALL_LIMIT','100')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
import torch
torch.set_num_threads(2)
import gradio as gr
import uvicorn
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from api.main import create_app
from api.model_service import ModelService
from api.speech_main import app as speech_app
from api.public_guard import PublicGuard

ROOT=Path(__file__).resolve().parent
service=ModelService()
text_app=create_app(service)
@asynccontextmanager
async def lifespan(app):
    async with text_app.router.lifespan_context(text_app):yield

app=FastAPI(title='Emotion Mirror · CBS4702',lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
app.add_middleware(PublicGuard)
# Include routes, rather than mounting /api twice; no internal API origins required.
for route in text_app.routes:
    if getattr(route,'path','') in ('/health','/models','/predict'):
        app.add_api_route('/api'+route.path,route.endpoint,methods=list(route.methods),response_model=route.response_model)
for route in speech_app.routes:
    if getattr(route,'path','') in ('/transcribe','/multimodal','/multimodal/health','/health'):
        path='/speech-health'if route.path=='/health'else route.path
        app.add_api_route('/api'+path,route.endpoint,methods=list(route.methods))

@app.get('/api/evidence')
def evidence():
    return {name:json.loads((ROOT/'web/data'/file).read_text())for name,file in [('ledger','ledger.json'),('errors','error_analysis.json')]}

with gr.Blocks(title='Emotion Mirror · service inspector')as inspector:
    gr.Markdown('# Emotion Mirror\n[Open the voice-first mirror](/)\n\nIndependent English text, expression and acoustic classifiers. No hidden-feeling detector.')
    gr.Markdown('The public demo processes recordings and snapshots in memory. Optional DeepSeek sends only selected text to its provider. Models and validation notes are linked from the Space repository. Open the direct `.hf.space` address for microphone and camera permissions.')
    status=gr.JSON(label='Loaded text models')
    refresh=gr.Button('Check model readiness')
    refresh.click(fn=service.health,outputs=status,api_name=False)
app=gr.mount_gradio_app(app,inspector,path='/service',footer_links=[],blocked_paths=[str(ROOT/'models'),str(ROOT/'results'),str(ROOT/'.env')],max_file_size='1mb')
app.mount('/',StaticFiles(directory=str(ROOT/'web/out'),html=True),name='website')
if __name__=='__main__':uvicorn.run(app,host='0.0.0.0',port=int(os.getenv('PORT','7860')),access_log=False,limit_concurrency=12)
