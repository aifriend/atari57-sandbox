"""FastAPI sidecar for the Atari57 research console.

Serves the static prototype at `/` and exposes:

  GET  /api/checkpoints       — list .ckpt files with parsed metadata
  GET  /api/games             — 57 Atari games + has_checkpoint flag
  GET  /api/algorithms        — 20 deep_rl_zoo algorithms grouped by family
  GET  /api/runs              — list tensorboard runs in runs/

Phase-2+ endpoints (eval/training/comparison/recordings) live alongside.

Launch:
  python -m frontend.server         # binds 127.0.0.1:8000

This module deliberately doesn't import deep_rl_zoo — keeps API healthy
even if the RL stack has import errors. Subprocesses do the heavy lifting.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .catalog import ALGORITHMS, GAMES_57, family_label

# Repo root = parent of frontend/. All paths are resolved relative to it.
REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RUNS_DIR = REPO_ROOT / "runs"
RECORDINGS_DIR = REPO_ROOT / "recordings"


app = FastAPI(
    title="Atari57 Research Console API",
    description="Sidecar that wires the frontend prototype to deep_rl_zoo.",
    version="0.1.0",
)

# Permissive CORS so the prototype can be served standalone during dev.
# Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────── helpers ─────────

# Checkpoint filenames in this codebase: <ALGO>_<Game>_<iter>.ckpt
# e.g. IQN_Pong_2.ckpt, PPO-RND_MontezumaRevenge_2.ckpt, PER-DQN_Pong_4.ckpt
CKPT_RE = re.compile(r"^(?P<algo>[A-Za-z0-9-]+)_(?P<game>[A-Za-z0-9]+)_(?P<iter>\d+)\.ckpt$")

# Map filename-prefix → catalog module name (filenames don't always match the
# deep_rl_zoo module path 1:1: "PER-DQN" ↔ prioritized_dqn, "PPO-RND" ↔ ppo_rnd).
ALGO_FILENAME_TO_MODULE = {
    "DQN": "dqn",
    "DOUBLE-DQN": "double_dqn",
    "PER-DQN": "prioritized_dqn",
    "PRIORITIZED-DQN": "prioritized_dqn",
    "DRQN": "drqn",
    "R2D2": "r2d2",
    "NGU": "ngu",
    "AGENT57": "agent57",
    "C51-DQN": "c51_dqn",
    "C51": "c51_dqn",
    "RAINBOW": "rainbow",
    "QR-DQN": "qr_dqn",
    "IQN": "iqn",
    "REINFORCE": "reinforce",
    "REINFORCE-B": "reinforce_baseline",
    "ACTOR-CRITIC": "actor_critic",
    "A2C": "a2c",
    "SAC": "sac",
    "PPO": "ppo",
    "PPO-ICM": "ppo_icm",
    "PPO-RND": "ppo_rnd",
    "IMPALA": "impala",
}


@dataclass
class CheckpointMeta:
    filename: str
    algo_module: str | None
    algo_label: str
    game: str
    iter: int
    size_bytes: int
    mtime: float


def _parse_checkpoint(p: Path) -> CheckpointMeta | None:
    m = CKPT_RE.match(p.name)
    if m is None:
        return None
    algo_label = m.group("algo")
    return CheckpointMeta(
        filename=p.name,
        algo_module=ALGO_FILENAME_TO_MODULE.get(algo_label.upper()),
        algo_label=algo_label,
        game=m.group("game"),
        iter=int(m.group("iter")),
        size_bytes=p.stat().st_size,
        mtime=p.stat().st_mtime,
    )


def _list_checkpoints() -> list[CheckpointMeta]:
    if not CHECKPOINTS_DIR.exists():
        return []
    out: list[CheckpointMeta] = []
    for p in sorted(CHECKPOINTS_DIR.glob("*.ckpt")):
        parsed = _parse_checkpoint(p)
        if parsed is not None:
            out.append(parsed)
    return out


# ───────── API endpoints ─────────


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "repo_root": str(REPO_ROOT),
        "checkpoints_dir_exists": CHECKPOINTS_DIR.exists(),
        "runs_dir_exists": RUNS_DIR.exists(),
        "recordings_dir_exists": RECORDINGS_DIR.exists(),
    }


@app.get("/api/checkpoints")
def list_checkpoints() -> dict[str, Any]:
    items = [meta.__dict__ for meta in _list_checkpoints()]
    return {"count": len(items), "items": items}


@app.get("/api/games")
def list_games() -> dict[str, Any]:
    """57 Atari games + which ones have at least one bundled checkpoint."""
    checkpoints = _list_checkpoints()
    games_with_ckpt = {ckpt.game for ckpt in checkpoints}
    items = [
        {
            "name": name,
            "code": code,
            "has_checkpoint": name in games_with_ckpt,
        }
        for name, code in GAMES_57
    ]
    return {"count": len(items), "items": items}


@app.get("/api/algorithms")
def list_algorithms() -> dict[str, Any]:
    """20 deep_rl_zoo algorithms grouped by family, with checkpoint count."""
    checkpoints = _list_checkpoints()
    by_module: dict[str, list[CheckpointMeta]] = {}
    for ckpt in checkpoints:
        if ckpt.algo_module:
            by_module.setdefault(ckpt.algo_module, []).append(ckpt)

    items = []
    for module, label, family in ALGORITHMS:
        ckpts = by_module.get(module, [])
        items.append(
            {
                "module": module,
                "label": label,
                "family": family,
                "family_label": family_label(family),
                "checkpoint_count": len(ckpts),
                "games_with_checkpoint": sorted({c.game for c in ckpts}),
            }
        )
    return {"count": len(items), "items": items}


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    """Tensorboard runs in runs/ — directory names + event-file count."""
    if not RUNS_DIR.exists():
        return {"count": 0, "items": []}
    items = []
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        items.append(
            {
                "name": run_dir.name,
                "event_file_count": len(event_files),
                "size_bytes": sum(f.stat().st_size for f in event_files),
            }
        )
    return {"count": len(items), "items": items}


# ───────── eval streaming via WebSocket ─────────

# Client lifecycle:
#   1. open WS to /api/eval/stream
#   2. send {"action": "start", "checkpoint": "IQN_Pong_2.ckpt", "num_steps": 2000, "frame_stride": 2}
#   3. receive a stream of {"type": "init"|"step"|"episode"|"done"|"error", ...}
#   4. send {"action": "stop"} or close the socket — subprocess gets terminated.

@app.websocket("/api/eval/stream")
async def eval_stream(ws: WebSocket) -> None:
    await ws.accept()
    proc: asyncio.subprocess.Process | None = None
    pump_task: asyncio.Task | None = None

    async def pump_stdout(p: asyncio.subprocess.Process) -> None:
        """Read line-delimited JSON from subprocess stdout, forward to WS."""
        assert p.stdout is not None
        while True:
            line = await p.stdout.readline()
            if not line:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Subprocess emitted non-JSON (likely a deprecation warning); ignore.
                continue
            try:
                await ws.send_json(obj)
            except (RuntimeError, WebSocketDisconnect):
                return

    async def stop_proc() -> None:
        nonlocal proc, pump_task
        if pump_task and not pump_task.done():
            pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                pass
        if proc and proc.returncode is None:
            try:
                proc.send_signal(signal.SIGTERM)
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                proc.kill()
                await proc.wait()
        proc = None
        pump_task = None

    try:
        while True:
            msg = await ws.receive_json()
            action = msg.get("action")

            if action == "start":
                if proc is not None:
                    await ws.send_json({"type": "error", "message": "eval already running; stop first"})
                    continue

                ckpt_filename = msg.get("checkpoint")
                num_steps = int(msg.get("num_steps", 2000))
                frame_stride = int(msg.get("frame_stride", 2))

                if not ckpt_filename:
                    await ws.send_json({"type": "error", "message": "missing 'checkpoint' field"})
                    continue
                ckpt_path = CHECKPOINTS_DIR / ckpt_filename
                if not ckpt_path.exists() or ckpt_path.parent != CHECKPOINTS_DIR:
                    await ws.send_json({"type": "error", "message": f"checkpoint not found: {ckpt_filename}"})
                    continue
                meta = _parse_checkpoint(ckpt_path)
                if meta is None or meta.algo_module is None:
                    await ws.send_json({"type": "error", "message": f"can't parse algo from {ckpt_filename}"})
                    continue

                # Spawn the streaming eval subprocess.
                env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "frontend.stream_eval",
                    f"--checkpoint={ckpt_path}",
                    f"--algo={meta.algo_module}",
                    f"--game={meta.game}",
                    f"--num-steps={num_steps}",
                    f"--frame-stride={frame_stride}",
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                pump_task = asyncio.create_task(pump_stdout(proc))
                await ws.send_json(
                    {
                        "type": "started",
                        "checkpoint": ckpt_filename,
                        "algo": meta.algo_module,
                        "game": meta.game,
                        "pid": proc.pid,
                    }
                )

            elif action == "stop":
                await stop_proc()
                await ws.send_json({"type": "stopped"})

            else:
                await ws.send_json({"type": "error", "message": f"unknown action: {action!r}"})

    except WebSocketDisconnect:
        pass
    finally:
        await stop_proc()


# ───────── static frontend ─────────

# Mount last so /api/* routes win.
app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend",
)


# ───────── module entry point ─────────


def main() -> None:
    import uvicorn

    uvicorn.run(
        "frontend.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
