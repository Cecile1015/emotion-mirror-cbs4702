"""Locally build Render's static UI without changing the development checkout."""
from pathlib import Path
import os,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'.render-export-build'
def main():
    if BUILD.exists():shutil.rmtree(BUILD)
    shutil.copytree(ROOT/'web',BUILD,ignore=shutil.ignore_patterns('node_modules','.next*','out','.env*','*.tsbuildinfo','playwright-report','test-results'))
    shutil.rmtree(BUILD/'app/api');shutil.rmtree(BUILD/'tests',ignore_errors=True)
    (BUILD/'node_modules').symlink_to(ROOT/'web/node_modules',target_is_directory=True)
    shutil.copy2(ROOT/'deployment/render/next-export.config.mjs',BUILD/'next.config.mjs')
    node=shutil.which('node')or str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
    env=os.environ.copy();env['NEXT_TELEMETRY_DISABLED']='1'
    subprocess.run([node,str(BUILD/'node_modules/next/dist/bin/next'),'build','--webpack'],cwd=BUILD,env=env,check=True)
    out=ROOT/'web/out'
    if out.exists():shutil.rmtree(out)
    shutil.copytree(BUILD/'out',out)
    print('Ready: PORT=10000 .venv/bin/python -m uvicorn api.hosted:create_hosted_app --factory --host 127.0.0.1 --port 10000')
if __name__=='__main__':main()
