"""Download the pinned English transcription model; no emotion models are changed."""
from pathlib import Path
import hashlib
import json
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models/speech/tiny.en'
REVISION = '0d3d19a32d3338f10357c0889762bd8d64bbdeba'
FILES = ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt', 'README.md')
if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for name in FILES:
        path = DEST / name
        if not path.exists():
            url = f'https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/{REVISION}/{name}'
            temporary = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(url, timeout=90) as source, temporary.open('wb') as target:
                while chunk := source.read(1024 * 1024): target.write(chunk)
            temporary.replace(path)
        records.append({'file': name, 'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()})
        print(f'Ready: {name}', flush=True)
    (ROOT / 'docs/reproducibility/speech_model.json').write_text(json.dumps({'repository': 'Systran/faster-whisper-tiny.en', 'revision': REVISION, 'license': 'MIT', 'files': records}, indent=2) + '\n')
