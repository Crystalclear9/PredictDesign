from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CACHE_PATHS = (
    ROOT / ".pytest_cache",
    ROOT / ".tmp",
    ROOT / "predictdesign.egg-info",
)
SKIP_PARTS = {".git", ".venv"}


def _iter_pycache_dirs(root: Path) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("__pycache__"):
        if not path.is_dir():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        matches.append(path)
    return sorted(matches)


def _iter_smoke_result_dirs(root: Path) -> list[Path]:
    results_root = root / "results"
    if not results_root.exists():
        return []
    matches = []
    for path in results_root.iterdir():
        if path.is_dir() and path.name.startswith("parallel_api_test_smoke"):
            matches.append(path)
    return sorted(matches)


def _path_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total / 1024 / 1024


def _remove_path(path: Path, *, execute: bool) -> str:
    if not path.exists():
        return f"skip missing: {path}"
    size_mb = _path_size_mb(path)
    if not execute:
        return f"would remove: {path} ({size_mb:.2f} MB)"
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return f"removed: {path} ({size_mb:.2f} MB)"


def _archive_path(path: Path, archive_root: Path, *, execute: bool) -> str:
    if not path.exists():
        return f"skip missing: {path}"
    target = archive_root / path.name
    size_mb = _path_size_mb(path)
    if not execute:
        return f"would archive: {path} -> {target} ({size_mb:.2f} MB)"
    archive_root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(path), str(target))
    return f"archived: {path} -> {target} ({size_mb:.2f} MB)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean reproducible cache files and optionally archive old smoke-run outputs."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply the cleanup. Without this flag, the script only prints a dry run.",
    )
    parser.add_argument(
        "--archive-smoke-results",
        action="store_true",
        help="Move results/parallel_api_test_smoke* into results/archive/ instead of leaving them at the top level.",
    )
    args = parser.parse_args()

    operations: list[str] = []
    for path in CACHE_PATHS:
        operations.append(_remove_path(path, execute=args.execute))
    for path in _iter_pycache_dirs(ROOT):
        operations.append(_remove_path(path, execute=args.execute))

    if args.archive_smoke_results:
        archive_root = ROOT / "results" / "archive"
        for path in _iter_smoke_result_dirs(ROOT):
            operations.append(_archive_path(path, archive_root, execute=args.execute))

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"[cleanup] mode={mode}")
    for line in operations:
        print(line)


if __name__ == "__main__":
    main()

