# Atari57 Research Console — User Guide

A hands-on guide to using the live research console. If you want the API, the architecture, or the internals, read [frontend/README.md](frontend/README.md). This document only covers what you, the operator, see and click.

---

## 1. Launch

From the repo root:

```bash
./start.sh
```

The script kills anything on port 8000, starts the FastAPI sidecar, polls `/api/health`, and opens the browser the moment the server is up. Stop it with `Ctrl+C` — that tears down the server, the browser-poller, and any subprocess it spawned.

The page is a single 1600×1000 console that scales proportionally to your window — resize freely, nothing gets clipped.

---

## 2. The screen at a glance

```
┌──────────── TOP BAR ────────────────────────────────────────────────┐
│ ATARI57 │ EXP │ ALGO │ GAME │ SEED │ DEVICE │ FPS │   UPTIME │ TB   │
├─────────┬───────────────────────────────────────┬───────────────────┤
│ LEFT    │ CENTER                                │ RIGHT             │
│         │                                       │                   │
│ search  │ ┌── viewport (canvas) ──┐             │ 03 RUN STATUS     │
│         │ │  agent plays here     │             │ 04 HYPERPARAMS    │
│ 01      │ └───────────────────────┘             │ 05 NETWORK        │
│ AGENTS  │  transport: ◀ ▶ ▶ ↻  scrubber  speed  │ 06 COMPARISON     │
│ (algos) │                                       │ 07 EVENT LOG      │
│         │ 08 chart           │ 09 action panel  │                   │
│ 02      │ (episode return)   │ (action dist)    │                   │
│ 57 GAMES│                    │                  │                   │
│         │                    │                  │                   │
├─────────┴───────────────────────────────────────┴───────────────────┤
│  TRAIN status │ CKPT │ REPLAY │ GPU │ ticker (live tail)  │ pytorch │
└──────────── BOTTOM BAR ─────────────────────────────────────────────┘
```

Three columns. The **left** chooses what to run, the **center** is where the agent plays and the chart updates, the **right** is the inspector (mostly informational).

---

## 3. Choose what to run (left column)

**Panel 01 · AGENTS** — list of all 20 deep_rl_zoo algorithms grouped by family (policy / value / distributional). The number on each row is the count of bundled checkpoints for that algorithm. Click an algorithm to select it; the selection drives the top-bar `ALGO` cell and the right-side network diagram.

**Panel 02 · ATARI · 57 GAMES** — every Atari 2600 game. Games with at least one bundled checkpoint are highlighted (today: **Pong**, **MontezumaRevenge**). Click a game to select it; the top-bar `GAME` cell and the chart's title update.

> The bundled checkpoints today are **IQN/Pong**, **PER-DQN/Pong**, **Rainbow/Pong**, **PPO-RND/MontezumaRevenge**. ▶ Play needs a bundled checkpoint that matches your `(algo, game)` pair. ▶ TRAIN works for any pair.

The **search** strip at the top of the left column is decorative — it shows the currently active algorithm filter.

---

## 4. Watch an agent play (▶ Play)

1. Pick `(algo, game)` in the left column where a bundled checkpoint exists. Easiest: **Rainbow + Pong**.
2. Hit the **▶ Play** button on the transport bar (just below the viewport).
3. A WebSocket opens, a `frontend/stream_eval.py` subprocess starts, and the canvas begins receiving real ALE frames at the agent's chosen frame stride (default every 2nd frame).

While playing:

- **Viewport HUD** — top-left shows `P1 AGENT` score, top-right `CPU` score, and the bottom row shows `FRAME`, `EPISODE`, and last `REWARD`.
- **Action distribution** (bottom-right of the center column) — bars update with the agent's real action choices for this game.
- **Step counter / ε / γ** — under the transport bar, just informational.

Click ▶ again to pause/stop the stream. Closing the tab also tears the subprocess down.

The **transport bar** has step-back, step-forward, and reset buttons, and a **scrubber** that lets you jump within the recorded episode buffer (most recent eval). The speed selector (¼× ½× 1× 4× ∞) controls playback rate when you're scrubbing — it does not change the trained agent's behavior.

---

## 5. Train a model (▶ TRAIN 5K)

The big secondary button **▶ TRAIN 5K** spawns a real subprocess:

```
python -m deep_rl_zoo.<algo>.run_atari --num_train_steps=5000 --num_eval_steps=500
```

Workflow:

1. Pick `(algo, game)` in the left column. Any pair works — you don't need a bundled checkpoint.
2. Click **▶ TRAIN 5K**. A new entry appears in the **Event Log** (panel 07, right column) with a job id and live tail of the subprocess stdout.
3. The bottom bar's **TRAIN** cell flips to `running`. Wait for it to flip back to `exited` (or `failed`).
4. When the run exits, the **chart** automatically re-fetches tensorboard scalars from `runs/` so the new `episode_return` series appears.

Cancel a running job via the API (`POST /api/training/jobs/<id>/stop`) or by closing `start.sh` (which SIGTERMs everything).

> **5k steps is a smoke run, not real training.** Real training is millions of steps. Use this button to verify the pipeline; for actual research, drive `deep_rl_zoo` from the CLI (see the main README §4) and the chart will pick up the new scalars next time you select that game.

---

## 6. Compare every bundled agent (▶ COMPARE)

The **▶ COMPARE** button runs a 5000-step eval against **every bundled checkpoint** in parallel (~10s on M1 CPU) and renders the actual mean returns in **panel 06 · AGENT COMPARISON** (right column), sorted descending.

Use this when you've added a new checkpoint to `checkpoints/` and want to see where it lands relative to the others. The action is non-blocking — the rest of the UI stays responsive while the parallel evals run.

`cmp-status` in the panel header shows `eval: idle` / `running` / `done`.

---

## 7. Replay a saved game (▶ REPLAY)

The **▶ REPLAY** button overlays the most recent self-play MP4 from `recordings/` above the canvas. MP4s come from CLI eval runs:

```bash
python -m deep_rl_zoo.iqn.eval_agent --environment_name=Pong \
    --load_checkpoint_file=./checkpoints/IQN_Pong_2.ckpt \
    --num_iterations=1 --num_eval_steps=2000 --nouse_tensorboard
```

The WebSocket play path (▶ Play) does **not** write MP4s — it streams frames directly to the browser. So if ▶ REPLAY says "no recordings", run an eval from the CLI first, or wait until a TRAIN run finishes (each iteration writes one).

---

## 8. The right-column inspector

Five stacked panels. Most are informational; the only interactive one today is the comparison panel (driven by ▶ COMPARE).

**03 · RUN STATUS** — current step / return / eta and a key-value list of the active run's structural choices (replay type, actors/learners, batch size, target sync, distribution atoms, etc.). Reads `state.selected` after you click in the left column.

**04 · HYPERPARAMETERS** — `learning_rate`, `discount γ`, `ε-greedy`, `replay α/β`, `n-step`, `batch_size`, `target_period`, `grad_clip`, `frame_stack`. The sliders are **read-only display** in this build. To change values, pass them via the API:

```bash
curl -X POST http://127.0.0.1:8000/api/training/start \
  -H 'Content-Type: application/json' -d '{
    "algo": "rainbow", "game": "Pong", "num_train_steps": 10000,
    "extra_args": ["--learning_rate=0.0005", "--discount=0.99"]
  }'
```

The chart will pick up the new run automatically.

**05 · NETWORK** — ASCII diagram of the loaded model's architecture. Decorative — it shows a typical Rainbow-dueling tree, not the actually-loaded network.

**06 · AGENT COMPARISON** — populated by ▶ COMPARE. Each row is `algorithm · game · mean / max / min episode return` from a parallel eval.

**07 · EVENT LOG** — live tail of the most recent training run. Each line is `[timestamp] [job_id] <stdout>`. Use this to watch a TRAIN 5K run progress in real time.

---

## 9. The chart (center, bottom-left)

Five tabs across the top: **EPISODE RETURN** · **TD-LOSS** · **VALUE** · **ENTROPY** · **FPS**. Only the active tab is wired to real data — it pulls from `tbparse` over `runs/<name>/`.

When the chart can't find a run for the selected `(algo, game)`, it shows the upstream sample run for that game (if one exists). Otherwise it shows an empty plot.

The legend is for visual reference; what's actually drawn is the single series for the selected pair.

---

## 10. Top bar and bottom bar

**Top bar** (left to right):

- `EXP` — current experiment id (synthesized from your selection).
- `ALGO` — selected algorithm.
- `GAME` — selected game.
- `SEED` / `DEVICE` — cosmetic; runs are seeded from the API request body, device is whatever PyTorch picks.
- `FPS` — viewport render rate, not training rate.
- `UPTIME` — wall-clock since the page loaded.
- `TENSORBOARD :6006` — port reminder. Run `tensorboard --logdir=./runs` to launch it.

**Bottom bar:**

- `TRAIN` — `idle` / `running` / `exited` / `failed`.
- `CKPT` — the last checkpoint relevant to your selection.
- `REPLAY` — replay-buffer pressure (cosmetic in eval mode).
- `GPU` — device tag.
- **Ticker** — scrolling tail of recent training events.
- `PYTORCH` — installed torch version.

---

## 11. Common workflows

**A) Quickest demo — watch a trained agent win at Pong**

1. Click **Rainbow** in the algo rail.
2. Click **Pong** in the game grid.
3. Hit **▶ Play**.
4. Watch the agent score 14–6 over a few minutes.

**B) Smoke-test the training pipeline**

1. Click any `(algo, game)` pair — e.g. **DQN + Breakout**.
2. Hit **▶ TRAIN 5K**.
3. Watch the event log for the job id and tail.
4. When `TRAIN` flips to `exited`, the chart updates with your new `episode_return` series.

**C) See where a new checkpoint stands**

1. Drop a `.ckpt` file into `checkpoints/`.
2. Refresh the page.
3. Hit **▶ COMPARE**. Wait ~10s.
4. Read the sorted results in panel 06.

**D) Replay an old run**

1. From the CLI: `python -m deep_rl_zoo.<algo>.eval_agent ...` to write an MP4 under `recordings/`.
2. In the UI, hit **▶ REPLAY** to watch the most recent one.

---

## 12. What's static (don't be fooled)

The original prototype shipped with placeholder content. These cells are still cosmetic in this build — knowing this saves you from chasing ghost values:

- Top-bar `SEED 0x2A1F` and `DEVICE mps:0` — placeholders.
- Hyperparameter sliders (panel 04) — read-only display, not editable.
- Network architecture diagram (panel 05) — generic Rainbow-dueling tree, not the loaded model.
- Bottom-bar `REPLAY 874,332 / 1,000,000` — decorative.
- The `chart-tabs` other than EPISODE RETURN are visual; only the active tab is wired.

Everything else — left column selectors, the four ▶ buttons, the chart, the comparison panel, the event log, the top-bar `EXP/ALGO/GAME` cells, and the bottom-bar `TRAIN/CKPT` cells — reflects real backend state.

---

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| ▶ Play does nothing | No bundled checkpoint for the selected `(algo, game)`. | Pick a pair from §3, or click any pair and use ▶ TRAIN 5K instead. |
| ▶ REPLAY says no recording | The WebSocket play path doesn't write MP4s. | Run an eval from the CLI (see §7), or wait for a TRAIN run to finish. |
| Chart is empty | No `runs/<name>/` for the selected pair. | Run ▶ TRAIN 5K once for that pair; the chart re-fetches when the job exits. |
| Event log is silent | No training started yet. | Hit ▶ TRAIN 5K. |
| Page won't load | Port 8000 already in use, or venv not set up. | `start.sh` clears port 8000 automatically. If it still fails, `./setup.sh` first. |
| Browser didn't auto-open | `NO_BROWSER=1` is set, or the browser-poller couldn't reach `/api/health` in time. | Open `http://127.0.0.1:8000` manually. |
| Layout looks tiny | The window is too small — the console scales proportionally. | Resize the window larger; nothing is clipped, just shrunk. |

---

## 14. Where to look next

- **Add a checkpoint to the bundle** — drop the `.ckpt` into `checkpoints/`. The catalog (`frontend/catalog.py`) infers `(algo, game)` from the filename pattern `<Algo>_<Game>_<n>.ckpt`.
- **Train with custom hyperparameters** — see §11 in the main README, or the `extra_args` example in §8 above.
- **Hook a new algorithm to the WebSocket play path** — add a factory in `frontend/stream_eval.py` (`ALGO_FACTORIES`).
- **Wire a new UI panel** — add markup in `frontend/index.html`, fetch logic in `frontend/app.js`, and (if needed) a new endpoint in `frontend/server.py`. Tests live in `frontend/test_server.py`.

For the deep dive on `deep_rl_zoo` itself — algorithms, network architectures, distributed orchestration — see [UPSTREAM_README.md](UPSTREAM_README.md).
