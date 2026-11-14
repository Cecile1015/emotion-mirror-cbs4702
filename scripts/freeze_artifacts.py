"""Create or verify a local, credential-free snapshot before application work.

python scripts/freeze_artifacts.py
python scripts/freeze_artifacts.py --verify local_backups/<snapshot>/manifest.json

The archive includes saved models (including ignored BERT weights), analysis
outputs, cached splits and source. Re-downloadable GloVe caches, environments,
Git history and .env credentials are excluded explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("src", "api", "notebooks", "models", "results", "docs", "report", "tests", "scripts")
EXCLUDED_PARTS = {"__pycache__", ".ipynb_checkpoints", ".cache", ".DS_Store"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def files(root: Path) -> list[Path]:
    candidates = []
    for name in DIRECTORIES:
        candidates.extend((root / name).rglob("*"))
    candidates.extend((root / "data").glob("*.parquet"))
    candidates.extend((root / "data" / "handcrafted").rglob("*"))
    candidates.extend(root.glob("requirements*.txt"))
    candidates.extend(root / name for name in ("README.md", ".gitignore", ".env.example", "pytest.ini"))
    return sorted({p for p in candidates if p.is_file() and not p.is_symlink()
                   and not (set(p.relative_to(root).parts) & EXCLUDED_PARTS)
                   and (not p.name.startswith(".env") or p.name == ".env.example")})


def snapshot(root: Path, output: Path) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output / stamp
    destination.mkdir(parents=True, exist_ok=False)
    entries = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": digest(p)}
               for p in files(root)]
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest = {"created_utc": stamp, "git_head": git.stdout.strip() if git.returncode == 0 else None,
                "python": platform.python_version(), "platform": platform.platform(),
                "excluded": [".env and credentials", ".git", ".venv", "data/embeddings", "local_backups"],
                "files": entries}
    archive = destination / "artifacts.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for item in entries:
            tar.add(root / item["path"], arcname=item["path"], recursive=False)
    # Verify the actual archived bytes, not just the original files.
    with tarfile.open(archive, "r:gz") as tar:
        for item in entries:
            h = hashlib.sha256()
            with tar.extractfile(item["path"]) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != item["sha256"]:
                raise RuntimeError(f"Snapshot verification failed: {item['path']}")
    manifest["archive_sha256"] = digest(archive)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()})
    (destination / "environment.txt").write_text("\n".join(packages) + "\n")
    return {"directory": str(destination), "files": len(entries), "archive_bytes": archive.stat().st_size,
            "archive_sha256": manifest["archive_sha256"]}


def verify(root: Path, path: Path) -> list[str]:
    manifest = json.loads(path.read_text())
    return [item["path"] for item in manifest["files"]
            if not (root / item["path"]).is_file() or digest(root / item["path"]) != item["sha256"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        changed = verify(args.root, args.verify)
        print(json.dumps({"changed_or_missing": changed}, indent=2))
        raise SystemExit(bool(changed))
    print(json.dumps(snapshot(args.root, args.output or args.root / "local_backups"), indent=2))
