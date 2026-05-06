from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
VENDOR_ROOT = PROJECT_ROOT / "vendor"
RESULTS_ROOT = PROJECT_ROOT / "results"
ACG_NAP_ROOT = DATA_ROOT / "acg_nap"
MARBLE_ROOT = VENDOR_ROOT / "prefetch-kv-mas" / "benchmarks" / "marble"
