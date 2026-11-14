"""Build an allowlisted HF upload directory. Never copy .env, datasets or backups."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'.hf-deploy'
BUILD=ROOT/'.hf-export-build'

def copy(relative):
    source=ROOT/relative;target=DEST/relative
    if source.is_dir():
        shutil.copytree(source,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.DS_Store','*.part','.cache'))
    else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)

def main():
    DEST.mkdir(exist_ok=True)
    for directory in ['api','src','models/bert','models/multimodal','models/speech','web/data','web/licenses']:
        copy(directory)
    for file in ['models/bilstm/bilstm.pt','models/classical/bow_nb.joblib','models/classical/tfidf_lr.joblib','models/classical/improved_lr.joblib',
                 'results/metrics/temperature_improved_classical.json','results/metrics/temperature_bilstm.json','results/metrics/temperature_bert.json',
                 'results/app/ledger.json','results/llm/selection.json','results/llm/prompt_llm.json',
                 'requirements-api.txt','requirements-models.txt','requirements-speech.txt','requirements-multimodal.txt','web/THIRD_PARTY_NOTICES.md',
                 'docs/MULTIMODAL.md','docs/reproducibility/multimodal_models.json','docs/reproducibility/multimodal_validation.json']:
        copy(file)
    for name in ['README.md','requirements.txt','app.py']:
        shutil.copy2(ROOT/'deployment/huggingface'/name,DEST/name)
    # Keep website source with artifact; no dependency caches, local settings or route proxies are needed at runtime.
    for directory in ['app','components','lib','public']:
        shutil.copytree(ROOT/'web'/directory,DEST/'web'/directory,dirs_exist_ok=True)
    for file in ['package.json','pnpm-lock.yaml','tsconfig.json','next.config.mjs']:
        shutil.copy2(ROOT/'web'/file,DEST/'web'/file)
    if BUILD.exists():shutil.rmtree(BUILD)
    shutil.copytree(ROOT/'web',BUILD,ignore=shutil.ignore_patterns('node_modules','.next*','out','.env*','*.tsbuildinfo','playwright-report','test-results'))
    shutil.rmtree(BUILD/'app/api')
    shutil.rmtree(BUILD/'tests',ignore_errors=True)
    (BUILD/'node_modules').symlink_to(ROOT/'web/node_modules',target_is_directory=True)
    (BUILD/'next.config.mjs').write_text("export default {output:'export',poweredByHeader:false,images:{unoptimized:true}};\n")
    node=shutil.which('node')or str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
    env=os.environ.copy();env['NEXT_TELEMETRY_DISABLED']='1'
    subprocess.run([node,str(BUILD/'node_modules/next/dist/bin/next'),'build','--webpack'],cwd=BUILD,env=env,check=True)
    if (DEST/'web/out').exists():shutil.rmtree(DEST/'web/out')
    shutil.copytree(BUILD/'out',DEST/'web/out')
    rows=[]
    for p in sorted(DEST.rglob('*')):
        if not p.is_file():continue
        if p.name.startswith('.env')or any(x in p.parts for x in ['.git','node_modules','__pycache__']):raise RuntimeError('Unexpected private/cache file')
        with p.open('rb')as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        rows.append({'path':str(p.relative_to(DEST)),'bytes':p.stat().st_size,'sha256':digest})
    (ROOT/'docs/reproducibility/space_upload_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(f'Prepared {len(rows)} files / {sum(x["bytes"]for x in rows)/1024**2:.1f} MB at {DEST}')
if __name__=='__main__':main()
