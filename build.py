#!/usr/bin/env python3
"""Release driver for YaClip: build -> archive -> checksum -> publish.

Usage:
    python build.py build [archive] [checksum] [--variant cpu|cuda] [--dry-run]
    python build.py publish --tag vX.Y.Z [--prerelease]
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
APP_NAME = "yaclip"

_ARCH_MAP = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
_OS_MAP = {"linux": "linux", "darwin": "macos", "win32": "windows"}


def _version() -> str:
    """Read the package version straight out of pyproject.toml (no extra dep)."""
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def _host_os() -> str:
    return _OS_MAP.get(sys.platform, sys.platform)


def _host_arch() -> str:
    return _ARCH_MAP.get(platform.machine().lower(), platform.machine().lower())


def _artifact_name(version: str, os_name: str, arch: str, variant: str) -> str:
    return f"{APP_NAME}_{version}_{os_name}_{arch}_{variant}"


def cmd_build(args: argparse.Namespace) -> Path:
    """Run PyInstaller against yaclip.spec. Returns the onedir output path.

    Both --distpath (final onedir app) and --workpath (build cache) are pinned under
    dist/ so the whole build -- final output and scratch files alike -- lives in one
    gitignored, single-`rm -rf dist`-able directory. PyInstaller's own default for
    --workpath is a top-level ./build/, which would otherwise scatter outside dist/.
    """
    out_dir = DIST_DIR / APP_NAME
    work_dir = DIST_DIR / "work"
    print(f"[build] variant={args.variant} os={_host_os()} arch={_host_arch()}")
    if args.dry_run:
        print(
            f"[dry-run] would run: pyinstaller --noconfirm --distpath {DIST_DIR} "
            f"--workpath {work_dir} yaclip.spec"
        )
        return out_dir

    shutil.rmtree(out_dir, ignore_errors=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--distpath",
            str(DIST_DIR),
            "--workpath",
            str(work_dir),
            "yaclip.spec",
        ],
        cwd=ROOT,
        check=True,
    )
    return out_dir


def cmd_archive(args: argparse.Namespace) -> Path:
    """Zip the onedir build plus config.yaml.example/README/LICENSE into one archive."""
    version = _version()
    name = _artifact_name(version, _host_os(), _host_arch(), args.variant)
    archive_path = DIST_DIR / f"{name}.zip"
    src_dir = DIST_DIR / APP_NAME

    print(f"[archive] {archive_path.name}")
    if args.dry_run:
        print(f"[dry-run] would zip {src_dir} -> {archive_path}")
        return archive_path

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in src_dir.rglob("*"):
            if file.is_file():
                zf.write(file, Path(APP_NAME) / file.relative_to(src_dir))
    return archive_path


def cmd_checksum(args: argparse.Namespace) -> Path:
    """Write checksums.txt (SHA256) covering every .zip in dist/."""
    checksum_path = DIST_DIR / "checksums.txt"
    archives = sorted(DIST_DIR.glob("*.zip"))

    print(f"[checksum] {len(archives)} archive(s) -> {checksum_path.name}")
    if args.dry_run:
        for archive in archives:
            print(f"[dry-run] would hash {archive.name}")
        return checksum_path

    lines = []
    for archive in archives:
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        lines.append(f"{digest}  {archive.name}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def cmd_publish(args: argparse.Namespace) -> None:
    """Create/update a GitHub Release at --tag and upload every dist/ archive + checksums.txt."""
    if not args.tag:
        print("error: --tag is required for publish (e.g. --tag v0.1.0)", file=sys.stderr)
        sys.exit(1)

    assets = sorted(DIST_DIR.glob("*.zip")) + [DIST_DIR / "checksums.txt"]
    assets = [a for a in assets if a.exists()]
    if not assets:
        print("error: no archives found in dist/ -- run build/archive/checksum first", file=sys.stderr)
        sys.exit(1)

    gh_args = ["gh", "release", "create", args.tag, *[str(a) for a in assets], "--title", args.tag, "--notes", f"YaClip {args.tag}"]
    if args.prerelease:
        gh_args.append("--prerelease")

    print(f"[publish] gh release create {args.tag} ({len(assets)} asset(s))")
    if args.dry_run:
        print(f"[dry-run] would run: {' '.join(gh_args)}")
        return

    subprocess.run(gh_args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stages",
        nargs="*",
        default=["build", "archive", "checksum"],
        choices=["build", "archive", "checksum", "publish"],
        help="Stages to run in order (default: build archive checksum)",
    )
    parser.add_argument("--variant", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tag", default=None, help="Release tag (required for publish)")
    parser.add_argument("--prerelease", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions, do nothing")
    args = parser.parse_args()

    DIST_DIR.mkdir(exist_ok=True)

    dispatch = {
        "build": cmd_build,
        "archive": cmd_archive,
        "checksum": cmd_checksum,
        "publish": cmd_publish,
    }
    for stage in args.stages:
        dispatch[stage](args)


if __name__ == "__main__":
    main()
