"""FastAPI app entrypoint.

Run (from ~/pylibfranka-venv, the same venv franka-policy-runner validated
for running camera+robot+web all in one process):

    ~/pylibfranka-venv/bin/uvicorn franka_deploy.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

# The 1 kHz robot control thread (franka_deploy.robot.fr3_runtime) shares
# this process/GIL with the FastAPI event loop and the policy-predict
# thread. Python's default 5 ms GIL switch interval is long enough that a
# `communication_constraints_violation` reflex was observed live on
# connect -- shortening it reduces how long the control thread can be kept
# waiting for the GIL by any of that other work.
sys.setswitchinterval(0.0005)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from franka_deploy.api import routes_cameras, routes_config, routes_control, routes_status

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

app = FastAPI(title="franka-deploy")
app.include_router(routes_config.router)
app.include_router(routes_control.router)
app.include_router(routes_status.router)
app.include_router(routes_cameras.router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
