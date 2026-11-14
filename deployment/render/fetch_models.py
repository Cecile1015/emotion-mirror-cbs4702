"""Download public pinned artifacts at build time, validating every SHA-256.

No HF authentication is required. No secrets or remote model Python code are used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
MANIFEST=Path(__file__).with_name('model_downloads.json')
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def prepare(root,verify_only=False):
    for row in json.loads(MANIFEST.read_text()):
        path=root/row['path']
        if path.is_file() and digest(path)==row['sha256']:continue
        if verify_only:raise RuntimeError('Missing or mismatched artifact: '+row['path'])
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(path.suffix+'.download')
        try:
            with urllib.request.urlopen(row['url'],timeout=120)as r,temp.open('wb')as f:
                count=0
                while chunk:=r.read(1024*1024):
                    count+=len(chunk)
                    if count>row['bytes']:raise RuntimeError('Artifact exceeds expected size')
                    f.write(chunk)
            if temp.stat().st_size!=row['bytes'] or digest(temp)!=row['sha256']:raise RuntimeError('Artifact hash mismatch: '+row['path'])
            temp.replace(path)
            print('Verified:',row['path'],flush=True)
        finally:
            if temp.exists():temp.unlink()
    print('All public model files match the deployment manifest.',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args();prepare(args.root,verify_only=args.verify_only)
