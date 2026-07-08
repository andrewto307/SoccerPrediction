"""Make the project's `src/` importable and force offline defaults for tests."""

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Tests must never hit the network or require an API key.
os.environ.setdefault("LIVE_PROVIDER", "mock")
