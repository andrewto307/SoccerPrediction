"""Convenience launcher so the API runs from any working directory.

    ../myenv/bin/python src/serve.py     # from the project root (SoccerPrediction/)

Adds this src/ directory to sys.path so the `api` and `live` packages and the flat
modules (model_configs, data_preprocessing, ...) all resolve, then starts uvicorn.
Override HOST / PORT via environment variables if needed.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
        # Trust X-Forwarded-* from the reverse proxy (Caddy) so client IPs and
        # scheme are correct (used by the rate limiter and logging).
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
