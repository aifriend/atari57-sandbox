# Atari57 Research Console — Frontend

A live research-console UI for the Atari57 sandbox. Visualizes deep_rl_zoo agents playing Atari, drives real eval and training runs, and tails real tensorboard scalars.

![status: live](https://img.shields.io/badge/status-live-brightgreen)
![backend: FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)

## What's wired up

| Surface | Source of truth |
|---|---|
| 57-game grid | `GET /api/games` — bundled checkpoints flag the games as "trained" |
| Algorithm rail (20 algos) | `GET /api/algorithms` — checkpoint counts per algo |
| Game viewport (canvas) | `WebSocket /api/eval/stream` — real ALE frames from `frontend/stream_eval.py` |
| Action distribution panel | Per-action counts aggregated from the eval stream |
| Episode-return chart | `GET /api/runs/<name>/scalars` — real `episode_return` series via `tbparse` |
| Agent comparison panel | `POST /api/comparison/run` — parallel evals (5k steps each) on every bundled checkpoint |
| Replay overlay | `GET /api/recordings/...` — most recent self-play MP4 |
| ▶ TRAIN 5K button | `POST /api/training/start` — spawns `python -m deep_rl_zoo.<algo>.run_atari` for a 5k-step run |
| Top-bar EXP / ALGO / GAME labels | Synthesized from `state.selected` after the user clicks |

Originally designed in [Claude Design](https://claude.ai/design); the prototype's mock data has been progressively replaced by the FastAPI sidecar at `frontend/server.py`.

## Run it

The easiest path — one command from the repo root:

```bash
./start.sh
```

That runs the launcher script: frees the configured port (`PORT=8000` by default), cleans stale training-job logs, starts uvicorn with `--reload`, polls `/api/health`, and opens the browser when the server's up. Override port / host / no-browser via env:

```bash
PORT=8123 ./start.sh
HOST=0.0.0.0 ./start.sh
NO_BROWSER=1 ./start.sh
```

If you'd rather run uvicorn yourself:

```bash
# from the repo root, with .venv activated
PYTHONPATH=. python -m frontend.server                    # production-ish
PYTHONPATH=. uvicorn frontend.server:app --reload         # dev with hot reload
```

Open <http://127.0.0.1:8000> in a browser. Layout is fixed-width 1600px (research-console aesthetic, not responsive).

## Try the live features

```text
1. Pick an algorithm in the left rail and a game in the bottom-left grid.
2. ▶  on the transport — opens a WebSocket and streams real Atari frames
       from a deep_rl_zoo greedy actor running against the bundled
       checkpoint that matches your selection. Action distribution panel
       updates with the agent's real action choices. Score and frame
       counter update from the stream.
3. ▶ TRAIN 5K — spawns python -m deep_rl_zoo.<algo>.run_atari
       --num_train_steps=5000 as a subprocess. Status updates in the
       event log. When the run exits, the chart re-fetches scalars to
       pick up the new tensorboard events.
4. ▶ COMPARE — runs 5000-step evals against every bundled checkpoint in
       parallel (~10s on M1 CPU) and renders the actual mean episode
       returns in the agent comparison panel, sorted descending.
5. ▶ REPLAY — opens an overlay above the canvas playing the most recent
       self-play MP4 from recordings/.
```

## API reference

### Read-only

```
GET  /api/health
GET  /api/checkpoints           → 4 bundled .ckpt files with parsed metadata
GET  /api/games                 → 57 Atari games + has_checkpoint flag
GET  /api/algorithms            → 20 deep_rl_zoo algorithms grouped by family
GET  /api/runs                  → tensorboard runs in runs/
GET  /api/runs/{name}/scalars   → tbparse-parsed scalar series, downsampled
GET  /api/recordings            → MP4s under recordings/
GET  /api/recordings/{path}     → serves the MP4
```

### Eval streaming

```
WS   /api/eval/stream           → bidirectional. Client sends:
                                    {"action":"start", "checkpoint": "<file>", "num_steps": 5000, "frame_stride": 2}
                                  Server pushes:
                                    {"type":"init", ...}
                                    {"type":"step", "step": N, "frame_b64": "...", "reward": 0.0, "action": 2, "ep_return": 0.0}
                                    {"type":"episode", "episode_return": 14.0, "episode_steps": 2304}
                                    {"type":"done", "total_steps": 5000}
                                  Send {"action":"stop"} or close to terminate.
```

### Training control

```
POST /api/training/start        body: {"algo":"dqn","game":"Pong",
                                       "num_iterations":1,"num_train_steps":5000}
                                → {"job_id":"...", "pid":NNNN, "log_path":"..."}
GET  /api/training/jobs         → list with status (running / exited / failed)
POST /api/training/jobs/{id}/stop  → SIGTERM with 3s grace, then SIGKILL
GET  /api/training/jobs/{id}/log?tail=100  → last N log lines
```

### Agent comparison

```
POST /api/comparison/run        body: {"checkpoints":[...], "num_steps":5000}
                                → {results: [{checkpoint, algo, game, episodes, returns, mean, max, min}, ...]}
```

## Architecture

```
                                                ┌──────────────────────────────────┐
  Browser  ─────────────────────────────────►   │  FastAPI (frontend/server.py)    │
   index.html (single page, 1600px fixed)       │                                  │
   app.js   (vanilla JS, no toolchain)          │  ── /api/* read-only ──          │
                                                │     [tbparse]   [pathlib]        │
   ◄─ JSON ──────────────────────────  /api/*   │                                  │
                                                │  ── /api/eval/stream (WS) ──     │
   ◄─ JSON-per-line over WS ──────────  WS      │     spawn frontend/stream_eval   │
                                                │       (PyTorch + deep_rl_zoo)    │
                                                │     ↳ subprocess(stdout)         │
                                                │                                  │
                                                │  ── /api/training/start ──       │
                                                │     spawn python -m              │
                                                │       deep_rl_zoo.<algo>.run_*   │
                                                │                                  │
                                                │  ── /api/comparison/run ──       │
                                                │     gather() N stream_eval       │
                                                │       subprocesses, aggregate    │
                                                │                                  │
                                                │  ── static / ──                  │
                                                │     index.html, app.js           │
                                                └──────────────────────────────────┘
```

## What's still mocked

The right-hand side of the screen still has some prototype-static content that wasn't swapped in this round:

- Hyperparameter values panel (learning rate, gamma, etc.) — readonly, pulled from typical defaults of the prototype, not from any live network. Editing them is the obvious next experiment; would map to extra `--learning_rate=...` args sent to `/api/training/start`.
- Model architecture diagram (the dueling Rainbow tree on the right) — pure ASCII layout, doesn't reflect the actually-loaded network.
- Top-bar `SEED 0x2A1F` / `DEVICE mps:0` cells — cosmetic; runs are seeded from `int(body.get('seed', 1))` on the backend.

## Tests

```bash
PYTHONPATH=. pytest frontend/test_server.py -v
# 18 tests, ~0.5s
```

The tests cover all read-only endpoints, validation paths for control endpoints (training start, comparison), path-traversal hardening on `/api/runs/{name}/scalars` and `/api/recordings/{path}`, and that the static HTML + JS still serve. Subprocess-spawning code paths are not exercised in CI — only the validation that surrounds them.
