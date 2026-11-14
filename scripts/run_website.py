"""Start the local website, emotion API and Whisper sidecar; Ctrl+C stops owned processes."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[1]

def response(url):
    try:
        with urllib.request.urlopen(url,timeout=2) as result:return json.load(result)
    except Exception:return None

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--production',action='store_true');args=parser.parse_args()
    python=ROOT/'.venv/bin/python'
    node=shutil.which('node')
    if not node:
        bundled=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
        if bundled.is_file():node=str(bundled)
    if not python.exists():raise SystemExit('Project .venv missing. Follow docs/WEBSITE.md first.')
    if not node:raise SystemExit('Install Node.js 22+ from https://nodejs.org, then reopen this launcher.')
    if not (ROOT/'web/node_modules/next/dist/bin/next').exists():raise SystemExit('Install web dependencies first: cd web && pnpm install --frozen-lockfile')
    if not (ROOT/'models/speech/tiny.en/model.bin').exists():raise SystemExit('Prepare voice input first: .venv/bin/python scripts/prepare_speech.py')
    if not (ROOT/'models/multimodal/face/model.safetensors').exists() or not (ROOT/'models/multimodal/acoustic/pytorch_model.bin').exists():
        print('Bonus models missing. Voice/text remain available; run scripts/prepare_multimodal.py to enable bonus inference.',flush=True)
    if args.production and not (ROOT/'web/.next/BUILD_ID').exists():raise SystemExit('Build the website first: cd web && pnpm build')
    env=os.environ.copy();env.update({'PATH':str(Path(node).parent)+os.pathsep+env.get('PATH',''),'NEXT_TELEMETRY_DISABLED':'1','PYTHONDONTWRITEBYTECODE':'1'})
    owned=[]
    try:
        configs=[(8000,'api.main:app','/models','models'),(8001,'api.speech_main:app','/health','model')]
        for port,app,path,key in configs:
            url=f'http://127.0.0.1:{port}{path}'; existing=response(url)
            if isinstance(existing,dict) and key in existing:
                
                if port==8001 and not response(f'http://127.0.0.1:{port}/multimodal/health'):
                    raise RuntimeError('An older speech service is running on port 8001. Stop the previous launcher with Ctrl+C, then start again to enable Mind-reading.')
                print(f'Reusing service on port {port}.',flush=True);continue
            process=subprocess.Popen([str(python),'-m','uvicorn',app,'--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=env)
            owned.append(process)
            for _ in range(120):
                if process.poll() is not None:raise RuntimeError(f'Service on port {port} did not start. Check for a port conflict.')
                if response(url):break
                time.sleep(.5)
            else:raise RuntimeError(f'Service on port {port} did not become ready.')
        command=[node,str(ROOT/'web/node_modules/next/dist/bin/next')]
        command+=['start'] if args.production else ['dev','--webpack']
        command+=['--hostname','127.0.0.1','--port','3000']
        web=subprocess.Popen(command,cwd=ROOT/'web',env=env);owned.append(web)
        print('\nOpen http://127.0.0.1:3000 in Chrome. Keep this window open.\nPress Ctrl+C to stop services started by this launcher.\n',flush=True)
        web.wait()
    except KeyboardInterrupt:
        print('\nStopping the website…',flush=True)
    finally:
        for process in reversed(owned):
            if process.poll() is None:process.terminate()
        for process in reversed(owned):
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:process.kill()

if __name__=='__main__':main()
