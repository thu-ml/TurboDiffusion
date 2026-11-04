"""Download revision-pinned H3 weights and verify their manifest checksums."""
from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path, PurePosixPath

from huggingface_hub import snapshot_download

DMD_REPO = "TurboDiffusion/TurboH3-FL2VA-I2VA"
DMD_REVISION = "60266aa6c44e2ff371bc6aa89af8437ed9494507"
COMPONENT_REPO = "MiniMaxAI/MiniMax-H3"
COMPONENT_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DMD_PATTERNS = (
    "transformer/*", "w8a8/*", "adaln/*", "README.md", "LICENSE", "NOTICE", "manifest.json",
)
ADALN_PATTERNS = ("adaln/*.safetensors", "adaln/*.json", "manifest.json")
COMPONENT_PATTERNS = tuple(
    f"FL2VA/{name}/*"
    for name in ("text_encoder", "processor", "tokenizer", "video_vae", "audio_vae")
)


def verify_manifest(root: Path, patterns: tuple[str, ...]) -> int:
    entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["files"]
    selected = {
        name: record for name, record in entries.items()
        if any(fnmatchcase(name, pattern) for pattern in patterns)
    }
    for pattern in patterns:
        if pattern != "manifest.json" and not any(fnmatchcase(name, pattern) for name in selected):
            raise ValueError(f"manifest has no files matching {pattern!r}")
    for name, record in selected.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {name!r}")
        path = root / name
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(
                f"missing or incorrect-size model file: {path}; "
                "remove any local copy and rerun the downloader"
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}; remove it and rerun the downloader")
    return len(selected)


def download(
    dit_dir: Path, *, components_dir: Path | None = None,
    adaln_only: bool = False, skip_w8a8: bool = False,
) -> None:
    patterns = ADALN_PATTERNS if adaln_only else DMD_PATTERNS
    if skip_w8a8:
        patterns = tuple(pattern for pattern in patterns if pattern != "w8a8/*")
    snapshot_download(
        repo_id=DMD_REPO, revision=DMD_REVISION,
        allow_patterns=list(patterns), local_dir=str(dit_dir),
    )
    count = verify_manifest(dit_dir, patterns)
    print(f"Verified {count} DMD files at {DMD_REVISION}", flush=True)
    if components_dir is not None:
        snapshot_download(
            repo_id=COMPONENT_REPO, revision=COMPONENT_REVISION,
            allow_patterns=list(COMPONENT_PATTERNS), local_dir=str(components_dir),
        )
        print(f"Downloaded official H3 components at {COMPONENT_REVISION}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-dir", type=Path, required=True)
    parser.add_argument("--components-dir", type=Path)
    parser.add_argument("--adaln-only", action="store_true")
    parser.add_argument("--skip-w8a8", action="store_true",
                        help="download BF16 weights without the published W8A8 export")
    args = parser.parse_args()
    download(
        args.dit_dir, components_dir=args.components_dir,
        adaln_only=args.adaln_only, skip_w8a8=args.skip_w8a8,
    )


if __name__ == "__main__":
    main()
