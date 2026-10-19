#!/usr/bin/env python3

import argparse
import json
import logging
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.eot_streaming_analysis import (
    StreamingState,
    export_core_csvs,
    run_shard,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

DEFAULT_PREFIX = "s3://eotarchive/eot-index/collections/EOT-2024/indexes/*.gz"


def parse_s5cmd_listing(output: str) -> list[str]:
    names: list[str] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[-1].split("/")[-1]
        if name.endswith(".cdxj.gz"):
            names.append(name)
    return sorted(set(names))


def parse_s5cmd_sizes(output: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[-1].split("/")[-1]
        if name.endswith(".cdxj.gz"):
            sizes[name] = int(parts[-2])
    return sizes


def list_shards(prefix: str) -> list[str]:
    result = subprocess.run(
        ["s5cmd", "--no-sign-request", "ls", prefix],
        check=True,
        capture_output=True,
        text=True,
    )
    root = prefix.rsplit("/", 1)[0]
    return [f"{root}/{name}" for name in parse_s5cmd_listing(result.stdout)]


def list_shard_sizes(prefix: str) -> dict[str, int]:
    result = subprocess.run(
        ["s5cmd", "--no-sign-request", "ls", prefix],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_s5cmd_sizes(result.stdout)


def load_manifest(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read manifest {path}: {error}") from error


def save_manifest(path: Path, manifest: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def stage_shard(
    remote_path: str, staging_dir: Path, expected_size: int | None = None
) -> Path:
    staging_dir.mkdir(parents=True, exist_ok=True)
    local_path = staging_dir / Path(remote_path).name
    if (
        local_path.exists()
        and expected_size is not None
        and local_path.stat().st_size != expected_size
    ):
        local_path.unlink()
    if not local_path.exists():
        partial_path = local_path.with_suffix(local_path.suffix + ".part")
        if partial_path.exists():
            partial_path.unlink()
        subprocess.run(
            ["s5cmd", "--no-sign-request", "cp", remote_path, str(partial_path)],
            check=True,
        )
        if expected_size is not None and partial_path.stat().st_size != expected_size:
            actual_size = partial_path.stat().st_size
            partial_path.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded {partial_path} has size {actual_size}; "
                f"expected {expected_size}."
            )
        partial_path.replace(local_path)
    return local_path


def state_has_shard(state_path: Path, shard_name: str) -> bool:
    if not state_path.exists():
        return False
    connection = sqlite3.connect(state_path)
    try:
        return bool(
            connection.execute(
                "SELECT 1 FROM chunks WHERE shard = ? LIMIT 1", (shard_name,)
            ).fetchone()
        )
    except sqlite3.OperationalError:
        return False
    finally:
        connection.close()


def run_census(
    *,
    prefix: str,
    state_path: Path,
    staging_dir: Path,
    output_dir: Path,
    batch_size: int,
    max_shards: int | None,
    delete_after_success: bool,
    exact_distinct: bool,
) -> None:
    shards = list_shards(prefix)
    shard_sizes = list_shard_sizes(prefix)
    if max_shards is not None:
        shards = shards[:max_shards]
    if not shards:
        raise ValueError(f"No raw CDXJ shards found for {prefix}")

    configuration_state = StreamingState(
        state_path,
        exact_distinct=exact_distinct,
        batch_size=batch_size,
    )
    configuration_state.close()

    manifest_path = state_path.with_name("manifest.json")
    manifest = load_manifest(manifest_path)
    for index, remote_path in enumerate(shards, start=1):
        shard_name = Path(remote_path).name
        existing = manifest.get(shard_name, {})
        if existing.get("status") == "complete" and state_has_shard(
            state_path, shard_name
        ):
            logger.info(
                "Skipping completed shard %s/%d: %s", index, len(shards), shard_name
            )
            continue

        logger.info("Staging shard %s/%d: %s", index, len(shards), shard_name)
        local_path = stage_shard(remote_path, staging_dir, shard_sizes.get(shard_name))
        logger.info("Processing shard: %s", shard_name)
        run_shard(
            local_path,
            state_path,
            batch_size,
            exact_distinct=exact_distinct,
        )
        manifest[shard_name] = {
            "status": "complete",
            "remote_path": remote_path,
            "local_path": str(local_path),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        save_manifest(manifest_path, manifest)
        if delete_after_success:
            local_path.unlink()
            logger.info("Deleted verified shard: %s", local_path)

    export_core_csvs(state_path, output_dir, exact_distinct=exact_distinct)
    logger.info("Exported aggregate outputs to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable, one-shard-at-a-time EOT file-type census."
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--state", type=Path, default=Path("/tmp/eot-census/state.sqlite")
    )
    parser.add_argument(
        "--staging-dir", type=Path, default=Path("/tmp/eot-census/current")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/eot-census/output")
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument(
        "--delete-after-success",
        action="store_true",
        help="Delete each local shard after its state checkpoint and manifest are saved.",
    )
    parser.add_argument(
        "--exact-distinct",
        action="store_true",
        help="Retain exact URL/digest tables; this is not recommended for the full archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    run_census(
        prefix=args.prefix,
        state_path=args.state,
        staging_dir=args.staging_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_shards=args.max_shards,
        delete_after_success=args.delete_after_success,
        exact_distinct=args.exact_distinct,
    )


if __name__ == "__main__":
    main()
