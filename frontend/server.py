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


@app.get("/api/runs/{run_name}/scalars")
def run_scalars(run_name: str, tag: str | None = None, max_points: int = 500) -> dict[str, Any]:
    """Return tensorboard scalar series for a run.

    Query params:
      tag         — optional, filter to a specific tag (e.g. performance(env_steps)/episode_return)
      max_points  — downsample dense series to this many points (default 500)

    Response: {"run": str, "tags": {tag: [{"step": int, "value": float}, ...], ...}}
    """
    import tbparse

    # Hardening: reject path traversal — run_name must be a single dir name.
    if "/" in run_name or ".." in run_name:
        return {"run": run_name, "tags": {}, "error": "invalid run_name"}
    run_dir = RUNS_DIR / run_name
    if not run_dir.is_dir():
        return {"run": run_name, "tags": {}, "error": "run not found"}

    reader = tbparse.SummaryReader(str(run_dir))
    df = reader.scalars
    if df is None or len(df) == 0:
        return {"run": run_name, "tags": {}}

    if tag is not None:
        df = df[df.tag == tag]

    out: dict[str, list[dict[str, float]]] = {}
    for tag_name, sub in df.groupby("tag"):
        # Drop NaNs, sort by step.
        sub = sub.dropna(subset=["value"]).sort_values("step")
        if len(sub) == 0:
            continue
        # Downsample if needed.
        if len(sub) > max_points:
            stride = max(1, len(sub) // max_points)
            sub = sub.iloc[::stride]
        out[str(tag_name)] = [
            {"step": int(row.step), "value": float(row.value)}
            for row in sub.itertuples(index=False)
        ]
    return {"run": run_name, "tags": out}


# ───────── training subprocess control ─────────

# In-memory job table. Surviving across restarts isn't a goal — for a single-machine
# research console, this is plenty. Each job is a subprocess of `python -m
# deep_rl_zoo.<algo>.run_atari` with optional hyperparameter overrides.

@dataclass
class TrainingJob:
    job_id: str
    algo: str
    game: str
    pid: int
    started_at: float
    proc: asyncio.subprocess.Process | None = None
    args: list[str] = None  # type: ignore[assignment]
    log_path: Path | None = None

    def status(self) -> str:
        if self.proc is None:
            return "unknown"
        if self.proc.returncode is None:
            return "running"
        return "exited" if self.proc.returncode == 0 else "failed"


JOBS: dict[str, TrainingJob] = {}


@app.post("/api/training/start")
async def training_start(body: dict[str, Any]) -> dict[str, Any]:
    """Spawn `python -m deep_rl_zoo.<algo>.run_atari` with optional overrides.

    Body: {
      "algo": "dqn",
      "game": "Pong",
      "num_iterations": 1,
      "num_train_steps": 5000,
      "num_eval_steps": 1000,
      "extra_args": ["--learning_rate=0.0005"]   // optional, passed straight through
    }
    """
    import time
    import uuid

    algo = body.get("algo")
    game = body.get("game")
    if not algo or not game:
        return {"error": "algo and game are required"}

    # Validate algo against catalog.
    if not any(a[0] == algo for a in ALGORITHMS):
        return {"error": f"unknown algo: {algo}"}

    job_id = uuid.uuid4().hex[:8]
    log_path = REPO_ROOT / "runs" / f"_training_{job_id}.log"
    log_path.parent.mkdir(exist_ok=True)

    # Note: deep_rl_zoo's run_atari uses absl-py FLAGS, so booleans are toggled
    # via --use_tensorboard / --nouse_tensorboard (no =value). Tensorboard
    # defaults to True, so we leave it on by omitting the flag — the chart
    # endpoint reads its event files.
    args = [
        sys.executable, "-m", f"deep_rl_zoo.{algo}.run_atari",
        f"--environment_name={game}",
        f"--num_iterations={int(body.get('num_iterations', 1))}",
        f"--num_train_steps={int(body.get('num_train_steps', 5000))}",
        f"--num_eval_steps={int(body.get('num_eval_steps', 1000))}",
    ]
    for arg in body.get("extra_args") or []:
        if isinstance(arg, str) and arg.startswith("--"):
            args.append(arg)

    log_fh = log_path.open("wb")
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        stdout=log_fh,
        stderr=asyncio.subprocess.STDOUT,
    )
    job = TrainingJob(
        job_id=job_id,
        algo=algo,
        game=game,
        pid=proc.pid,
        started_at=time.time(),
        proc=proc,
        args=args[2:],  # skip "python -m"
        log_path=log_path,
    )
    JOBS[job_id] = job
    return {
        "job_id": job_id,
        "pid": proc.pid,
        "algo": algo,
        "game": game,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "started_at": job.started_at,
    }


@app.get("/api/training/jobs")
def training_jobs() -> dict[str, Any]:
    items = []
    for job in JOBS.values():
        items.append(
            {
                "job_id": job.job_id,
                "algo": job.algo,
                "game": job.game,
                "pid": job.pid,
                "started_at": job.started_at,
                "status": job.status(),
                "returncode": job.proc.returncode if job.proc else None,
                "log_path": str(job.log_path.relative_to(REPO_ROOT)) if job.log_path else None,
            }
        )
    items.sort(key=lambda j: j["started_at"], reverse=True)
    return {"count": len(items), "items": items}


@app.post("/api/training/jobs/{job_id}/stop")
async def training_stop(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        return {"error": "job not found"}
    if job.proc is None or job.proc.returncode is not None:
        return {"job_id": job_id, "status": job.status()}
    try:
        job.proc.send_signal(signal.SIGTERM)
        await asyncio.wait_for(job.proc.wait(), timeout=3.0)
    except (ProcessLookupError, asyncio.TimeoutError):
        try:
            job.proc.kill()
            await job.proc.wait()
        except ProcessLookupError:
            pass
    return {"job_id": job_id, "status": job.status(), "returncode": job.proc.returncode}


@app.get("/api/training/jobs/{job_id}/log")
def training_log(job_id: str, tail: int = 100) -> dict[str, Any]:
    """Return the last N lines of the job's log file (defaults to 100)."""
    job = JOBS.get(job_id)
    if job is None or job.log_path is None or not job.log_path.exists():
        return {"job_id": job_id, "lines": []}
    text = job.log_path.read_text(errors="replace")
    lines = text.splitlines()
    if tail > 0:
        lines = lines[-tail:]
    return {"job_id": job_id, "lines": lines}


# ───────── recordings (self-play MP4s) ─────────


@app.get("/api/recordings")
def list_recordings() -> dict[str, Any]:
    """List MP4s under recordings/. Each subdir is one recording session."""
    if not RECORDINGS_DIR.exists():
        return {"count": 0, "items": []}
    items = []
    for p in sorted(RECORDINGS_DIR.rglob("*.mp4")):
        try:
            rel = p.relative_to(RECORDINGS_DIR)
        except ValueError:
            continue
        items.append(
            {
                "filename": str(rel),  # e.g. IQN-greedy_PongNoFrameskip-v4_..../rl-video-episode-0.mp4
                "size_bytes": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "url": f"/api/recordings/{rel}",
            }
        )
    items.sort(key=lambda r: r["mtime"], reverse=True)
    return {"count": len(items), "items": items}


@app.get("/api/recordings/{path:path}")
def serve_recording(path: str) -> FileResponse:
    """Serve an MP4 by relative path under recordings/."""
    if ".." in path:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="invalid path")
    target = (RECORDINGS_DIR / path).resolve()
    if not target.exists() or not str(target).startswith(str(RECORDINGS_DIR.resolve())):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(target), media_type="video/mp4")


# ───────── agent comparison (parallel eval) ─────────


@app.post("/api/comparison/run")
async def comparison_run(body: dict[str, Any]) -> dict[str, Any]:
    """Run a quick eval against multiple bundled checkpoints in parallel.

    Body: {
      "checkpoints": ["Rainbow_Pong_2.ckpt", "PER-DQN_Pong_4.ckpt", ...],
      "num_steps": 1000          // defaults to 1000
    }

    Returns: {results: [{checkpoint, episodes, returns: [...], mean, max, min}, ...]}
    """
    checkpoints = body.get("checkpoints") or []
    num_steps = int(body.get("num_steps", 1000))
    if not checkpoints:
        return {"error": "checkpoints array is required"}

    async def eval_one(filename: str) -> dict[str, Any]:
        ckpt_path = CHECKPOINTS_DIR / filename
        if not ckpt_path.exists():
            return {"checkpoint": filename, "error": "not found"}
        meta = _parse_checkpoint(ckpt_path)
        if meta is None or meta.algo_module is None:
            return {"checkpoint": filename, "error": "can't parse algo from filename"}
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "frontend.stream_eval",
            f"--checkpoint={ckpt_path}",
            f"--algo={meta.algo_module}",
            f"--game={meta.game}",
            f"--num-steps={num_steps}",
            "--frame-stride=10000",  # effectively skip frames — only need episode events
            cwd=str(REPO_ROOT),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        episode_returns: list[float] = []
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "episode":
                episode_returns.append(float(obj.get("episode_return", 0.0)))
        await proc.wait()
        if not episode_returns:
            return {"checkpoint": filename, "algo": meta.algo_module, "game": meta.game, "episodes": 0, "returns": [], "mean": None}
        return {
            "checkpoint": filename,
            "algo": meta.algo_module,
            "game": meta.game,
            "episodes": len(episode_returns),
            "returns": episode_returns,
            "mean": sum(episode_returns) / len(episode_returns),
            "max": max(episode_returns),
            "min": min(episode_returns),
        }

    results = await asyncio.gather(*(eval_one(name) for name in checkpoints))
    return {"results": results, "num_steps": num_steps}


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
