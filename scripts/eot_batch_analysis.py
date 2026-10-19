#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path

DEFAULT_EOT_PREFIX = "s3://eotarchive/eot-index/collections/EOT-2024/indexes/*.gz"


def parse_s5cmd_listing(output: str) -> list[str]:
    files: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        filename = parts[-1].split("/")[-1]
        if filename.endswith(".gz") and "cluster.summary" not in filename.lower():
            files.append(filename)
    return files


def build_output_dir(output_root: Path, local_file: Path) -> Path:
    file_name = local_file.name
    if file_name.endswith(".cdxj.gz"):
        name = file_name[: -len(".cdxj.gz")]
    elif file_name.endswith(".gz"):
        name = file_name[: -len(".gz")]
    else:
        name = file_name
    return output_root / name


def list_remote_archives(prefix: str) -> list[str]:
    proc = subprocess.run(
        ["s5cmd", "--no-sign-request", "ls", prefix],
        capture_output=True,
        text=True,
        check=True,
    )
    remote_root = prefix.rsplit("/", 1)[0]
    return [
        f"{remote_root}/{filename}" for filename in parse_s5cmd_listing(proc.stdout)
    ]


def ensure_local_file(remote_name: str, staging_dir: Path) -> Path:
    local_path = staging_dir / Path(remote_name).name
    staging_dir.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        subprocess.run(
            ["s5cmd", "--no-sign-request", "cp", remote_name, str(local_path)],
            check=True,
        )
    return local_path


def analyze_local_file(local_file: Path, repo_root: Path, output_root: Path) -> Path:
    output_dir = build_output_dir(output_root, local_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.txt"
    if summary_path.exists():
        return output_dir

    analyzer = repo_root / "scripts" / "analyze_eot_file_types.py"
    subprocess.run(
        [
            "poetry",
            "run",
            "python",
            str(analyzer),
            "--input",
            str(local_file),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(repo_root),
        check=True,
    )
    return output_dir


def load_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        return {}
    try:
        contents = manifest_path.read_text(encoding="utf-8")
        if not contents.strip():
            return {}
        data = json.loads(contents)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {}


def save_manifest(manifest_path: Path, manifest: dict[str, dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_batch(
    *,
    prefix: str = DEFAULT_EOT_PREFIX,
    staging_dir: Path,
    output_root: Path,
    max_files: int | None = None,
    dry_run: bool = False,
) -> list[Path]:
    remote_files = list_remote_archives(prefix)
    if max_files is not None:
        remote_files = remote_files[:max_files]

    manifest_path = output_root / "batch_manifest.json"
    manifest = load_manifest(manifest_path)
    processed: list[Path] = []

    for remote_name in remote_files:
        remote_key = remote_name
        existing = manifest.get(remote_key, {})
        if existing.get("status") == "done":
            processed.append(Path(existing.get("output_dir", output_root)))
            continue

        local_file = ensure_local_file(remote_name, staging_dir)
        output_dir = build_output_dir(output_root, local_file)
        print(f"Processing {remote_name} -> {output_dir}")
        if dry_run:
            processed.append(output_dir)
            manifest[remote_key] = {
                "status": "done",
                "local_path": str(local_file),
                "output_dir": str(output_dir),
            }
            continue

        analyze_local_file(local_file, Path(__file__).resolve().parents[1], output_root)
        manifest[remote_key] = {
            "status": "done",
            "local_path": str(local_file),
            "output_dir": str(output_dir),
        }
        save_manifest(manifest_path, manifest)
        processed.append(output_dir)

    save_manifest(manifest_path, manifest)
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable batch analysis over the public EOT raw archive."
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_EOT_PREFIX,
        help="S3 prefix to list; defaults to the raw EOT-2024 index files.",
    )
    parser.add_argument(
        "--staging-dir",
        default="/tmp/eot-full-safe",
        help="Where to stage the remote .cdxj.gz files locally.",
    )
    parser.add_argument(
        "--output-root",
        default="/tmp/eot-full-output-safe",
        help="Root directory for per-file output folders.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for testing; processes only the first N remote files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List and plan work without running the file analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_batch(
        prefix=args.prefix,
        staging_dir=Path(args.staging_dir),
        output_root=Path(args.output_root),
        max_files=args.max_files,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
