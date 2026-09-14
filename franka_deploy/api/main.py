"""FastAPI app entrypoint.

Run (from ~/pylibfranka-venv, the same venv franka-policy-runner validated
for running camera+robot+web all in one process):

    ~/pylibfranka-venv/bin/uvicorn franka_deploy.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from franka_deploy.api import routes_config, routes_control, routes_status

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

app = FastAPI(title="franka-deploy")
app.include_router(routes_config.router)
app.include_router(routes_control.router)
app.include_router(routes_status.router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
