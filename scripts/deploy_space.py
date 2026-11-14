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
    args=parser.parse_args();api=HfApi();account=api.whoami()
    if not account.get('isPro'):raise SystemExit('Personal PRO must be active before creating this Docker Space. No subscription or hardware purchase is made by this script.')
    hardware='cpu-basic'
    if account['name']!=args.repo.split('/')[0]:raise SystemExit('The logged-in account must own the target Space.')
    if not (DEST/'Dockerfile').exists():raise SystemExit('Run scripts/prepare_space.py first.')
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
        api.create_repo(args.repo,repo_type='space',space_sdk='docker',space_hardware=hardware,private=False)
        prior.write_text(json.dumps({'repo':args.repo,'stage':'created','hardware_requested':hardware},indent=2)+'\n')
    api.add_space_variable(args.repo,key='EMOTION_DEEPSEEK_SESSION_CALL_LIMIT',value='100')
    api.add_space_variable(args.repo,key='EMOTION_CPU_THREADS',value='2')
    commit=api.upload_folder(repo_id=args.repo,repo_type='space',folder_path=str(DEST),ignore_patterns=['**/__pycache__/**','**/*.pyc','.env*'],commit_message='Deploy voice-first Emotion Mirror with independently validated optional bonus channels')
    info=api.space_info(args.repo)
    record={'repo':args.repo,'commit':commit.oid,'space_url':'https://huggingface.co/spaces/'+args.repo,'runtime':str(info.runtime.stage)if info.runtime else None,'hardware_requested':hardware,'deepseek_secret_requested':False,'validation':'Cloud smoke tests pending'}
    prior.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
