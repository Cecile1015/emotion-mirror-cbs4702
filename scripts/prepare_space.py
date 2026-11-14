"""Prepare a clean, allowlisted Docker Space source bundle; no local secrets or media."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '.hf-deploy'

def prepare():
    # This directory is generated only by this deployment script.
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir()
    paths = []
    for folder in ('api', 'src'):
        paths.extend(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).glob('*.py'))
    paths.extend(subprocess.check_output(['git', 'ls-files', '-z', 'web'], cwd=ROOT).decode().split('\0'))
    paths.extend([
        'requirements-models.txt', 'requirements-api.txt', 'requirements-speech.txt', 'requirements-multimodal.txt',
        'models/classical/bow_nb.joblib', 'models/classical/tfidf_lr.joblib', 'models/classical/improved_lr.joblib',
        'models/bilstm/bilstm.pt', 'models/bert/training_info.json',
        'results/metrics/temperature_improved_classical.json', 'results/metrics/temperature_bilstm.json',
        'results/metrics/temperature_bert.json', 'results/app/ledger.json',
        'results/llm/selection.json', 'results/llm/prompt_llm.json',
        'docs/MULTIMODAL.md', 'docs/reproducibility/multimodal_models.json',
        'docs/reproducibility/multimodal_validation.json',
        'docs/reproducibility/multimodal_yunet_validation.json', 'docs/reproducibility/face_detector.json',
        'docs/reproducibility/error_gallery.json', 'docs/reproducibility/face_detector_regression.json',
        'deployment/render/fetch_models.py', 'deployment/render/model_downloads.json',
        'deployment/render/next-export.config.mjs', 'deployment/render/start.py', '.dockerignore',
    ])
    for relative in sorted(set(filter(None, paths))):
        p = Path(relative)
        if p.name == ".env.example":
            continue
        if any(part in {'.git', 'node_modules', '__pycache__', '.next', '.next-dev', 'out'} for part in p.parts) or p.name.startswith('.env'):
            raise RuntimeError(f'Private/cache path rejected: {relative}')
        target = DEST / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / p, target)
    shutil.copy2(ROOT / 'deployment/render/Dockerfile', DEST / 'Dockerfile')
    shutil.copy2(ROOT / 'deployment/huggingface/README.md', DEST / 'README.md')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    (DEST / 'SOURCE.json').write_text(json.dumps({'github_repository': 'Cecile1015/emotion-mirror-cbs4702', 'base_commit': commit, 'deployment_overlay': 'Working-tree release: balanced 21-case gallery, Boo Darlings cover, YuNet face detector and validation; exact uploaded source hashes recorded in upload manifest'}, indent=2) + '\n')
    rows = []
    for p in sorted(DEST.rglob('*')):
        if not p.is_file():
            continue
        with p.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        rows.append({'path': p.relative_to(DEST).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest})
    (ROOT / 'docs/reproducibility/space_upload_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'Prepared {len(rows)} files / {sum(row["bytes"] for row in rows)/1024**2:.1f} MiB. No secrets copied.')

if __name__ == '__main__':
    prepare()
