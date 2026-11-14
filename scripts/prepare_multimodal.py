"""Fetch pinned, public bonus models. No remote Python code is executed."""
from pathlib import Path
import hashlib
import json
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'face': ('trpakov/vit-face-expression', 'ef0bc6fc34241b6587e7e009e7711357be28c024', ['config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md']),
    'acoustic': ('superb/wav2vec2-base-superb-er', '441a7599c3b22107314dcbd9166621c5c83f2cc5', ['config.json', 'preprocessor_config.json', 'pytorch_model.bin', 'README.md']),
}

def prepare():
    import sys
    sys.path.insert(0,str(ROOT))
    from deployment.render.fetch_models import prepare as fetch_pinned
    fetch_pinned(ROOT,only_prefix='models/multimodal/detector/')
    records = {}
    for channel, (repo, revision, files) in MODELS.items():
        dest = ROOT / 'models/multimodal' / channel
        dest.mkdir(parents=True, exist_ok=True)
        rows = []
        for name in files:
            path = dest / name
            if not path.exists():
                temp = path.with_suffix(path.suffix + '.part')
                with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', timeout=120) as source, temp.open('wb') as target:
                    while chunk := source.read(1024 * 1024): target.write(chunk)
                temp.replace(path)
            with path.open('rb') as handle: digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            rows.append({'file': name, 'bytes': path.stat().st_size, 'sha256': digest})
            print(f'Ready: {channel}/{name}', flush=True)
        records[channel] = {'repository': repo, 'revision': revision, 'license': 'Apache-2.0', 'files': rows}
    (ROOT / 'docs/reproducibility/multimodal_models.json').write_text(json.dumps(records, indent=2) + '\n')
if __name__ == '__main__': prepare()
