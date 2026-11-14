"""Upload only the reviewed deployment subset; use existing local HF login.

No token is printed, copied into the Space, or accepted on the command line.
No paid hardware is requested. An existing unrelated Space is never overwritten.
"""
from pathlib import Path
import argparse
import json
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'.hf-deploy'
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='Cecile1015/emotion-mirror-cbs4702')
    parser.add_argument('--include-deepseek-secret',action='store_true',help='Explicitly transfer the project key into this Space Secret.')
    args=parser.parse_args();api=HfApi();account=api.whoami()
    hardware='cpu-basic' if account.get('isPro') else 'zero-a10g'
    if account['name']!=args.repo.split('/')[0]:raise SystemExit('The logged-in account must own the target Space.')
    if not (DEST/'web/out/index.html').exists():raise SystemExit('Run scripts/prepare_space.py first.')
    manifest_path=ROOT/'docs/reproducibility/space_upload_manifest.json'
    import hashlib
    manifest=json.loads(manifest_path.read_text())
    actual={str(p.relative_to(DEST))for p in DEST.rglob('*')if p.is_file() and '__pycache__'not in p.parts}
    expected={r['path']for r in manifest}
    if actual!=expected:raise SystemExit('Deployment file list changed. Rebuild the reviewed subset.')
    for row in manifest:
        with (DEST/row['path']).open('rb')as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
        if digest!=row['sha256']:raise SystemExit('Deployment content changed. Rebuild the reviewed subset.')
    prior=ROOT/'docs/reproducibility/space_deployment.json'
    try:
        info=api.space_info(args.repo)
        if not prior.exists()or json.loads(prior.read_text()).get('repo')!=args.repo:raise SystemExit('Target already exists and is not this deployment. Choose a new Space name.')
    except RepositoryNotFoundError:
        api.create_repo(args.repo,repo_type='space',space_sdk='gradio',space_hardware=hardware,private=False)
        prior.write_text(json.dumps({'repo':args.repo,'stage':'created','hardware_requested':hardware},indent=2)+'\n')
    if args.include_deepseek_secret:
        from dotenv import dotenv_values
        import os
        secret=os.getenv('DEEPSEEK_API_KEY')or dotenv_values(ROOT/'.env').get('DEEPSEEK_API_KEY')
        if not secret:raise SystemExit('Project DeepSeek key is not configured.')
        api.add_space_secret(args.repo,key='DEEPSEEK_API_KEY',value=secret)
        print('DeepSeek configured as a Space Secret (value withheld).',flush=True)
    api.add_space_variable(args.repo,key='EMOTION_DEEPSEEK_SESSION_CALL_LIMIT',value='100')
    commit=api.upload_folder(repo_id=args.repo,repo_type='space',folder_path=str(DEST),ignore_patterns=['**/__pycache__/**','**/*.pyc','.env*'],commit_message='Deploy voice-first Emotion Mirror with independently validated optional bonus channels')
    info=api.space_info(args.repo)
    record={'repo':args.repo,'commit':commit.oid,'space_url':'https://huggingface.co/spaces/'+args.repo,'runtime':str(info.runtime.stage)if info.runtime else None,'hardware_requested':hardware,'deepseek_secret_requested':args.include_deepseek_secret,'validation':'Cloud smoke tests pending'}
    prior.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
