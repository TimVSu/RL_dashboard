"""App-level configuration.

Only server/runtime settings live here (where to find recorded runs, host,
port, playback tuning). Domain constants such as vehicle/station/operator
counts, rollout length, the no-assignment station index, and the H3
resolution must never be hard-coded here or anywhere in the app - they are
read per-run from each run's metadata.json (see dashboard/data/models.py).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RUNS_DIR = Path(
    os.environ.get("THESIS_DASHBOARD_RUNS_DIR", str(REPO_ROOT / "runs"))
).expanduser().resolve()

HOST = os.environ.get("THESIS_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("THESIS_DASHBOARD_PORT", "8050"))
DEBUG = os.environ.get("THESIS_DASHBOARD_DEBUG", "true").lower() in {"1", "true", "yes"}

# Playback: interval (ms) between timestep advances at 1x speed. Actual
# interval sent to dcc.Interval is this divided by the selected speed.
PLAYBACK_SPEEDS: list[float] = [0.5, 1, 2, 5]
DEFAULT_PLAYBACK_SPEED: float = 1.0
PLAYBACK_BASE_INTERVAL_MS: int = 600
