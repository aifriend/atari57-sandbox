# Atari57 Research Sandbox

[![tests](https://github.com/aifriend/atari57-sandbox/actions/workflows/test.yml/badge.svg)](https://github.com/aifriend/atari57-sandbox/actions/workflows/test.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

A ready-to-run setup of [michaelnny/deep_rl_zoo](https://github.com/michaelnny/deep_rl_zoo) for experimenting with deep RL algorithms on Atari games (PPO, DQN, Rainbow, IQN, R2D2, NGU, **Agent57**, and more).

> **Attribution.** All code under `deep_rl_zoo/` and `unit_tests/` is the original work of **Michael Hu** (`michaelnny`), released under Apache License 2.0. This repository is a **sandbox derivative** — it adds a `uv`-based setup script, a relaxed requirements pin set that builds on modern macOS (especially Apple Silicon), this README, a CI workflow, and a small set of corrections (one bundled checkpoint with an obsolete architecture removed). The upstream README is preserved as [`UPSTREAM_README.md`](UPSTREAM_README.md). See the [LICENSE](LICENSE) file for details.

> **Frontend prototype.** A hi-fi research-console UI for this project (CRT/phosphor aesthetic — game viewport, episode-return chart, action distribution, hyperparameter inspector, 57-game grid, replay scrubber) lives at [`frontend/index.html`](frontend/). It's a static, single-file HTML/CSS/JS prototype with mock data — open with `python -m http.server` from the `frontend/` directory. Designed in [Claude Design](https://claude.ai/design); see [`frontend/README.md`](frontend/README.md) for what it shows and what's faked.

> ⚠️ **About "play all 57 Atari games out of the box"**
>
> deep_rl_zoo is **research / educational code**. Upstream ships 5 pre-trained checkpoints; in this sandbox we kept 4 (Pong × 3, MontezumaRevenge × 1) — the bundled `PPO_Breakout_0.ckpt` was removed because its network architecture predates upstream commit `cd860e8` ("major update with breaking changes", June 2023) and no longer loads on current code (`Missing key(s) in state_dict: policy_head.0.weight, ...`).
>
> The upstream author explicitly notes agents were *"only tested on Atari Pong or Breakout, and we stop training once the agent has made some progress."* That shows in the bundled weights: see the eval scores in §2. There is **no community model zoo for deep_rl_zoo** that gives you trained Agent57/R2D2/NGU agents on all 57 games. To get more checkpoints you train them here yourself. See §5 for external sources of pre-trained Atari agents (different ecosystem, different setup).

---

## 1. Setup (one command)

```bash
./setup.sh
```

What this does:
- Installs `snappy` and `ffmpeg` via Homebrew if missing.
- Creates a Python 3.10 venv in `.venv/` with `uv` — **native arm64 on Apple Silicon** (uv downloads `cpython-3.10-macos-aarch64` rather than reusing the brew x86_64 build, which would run under Rosetta and be ~2× slower).
- Installs PyTorch 2.2+, gym 0.25.2 (with `[box2d]` extra so LunarLander works), ALE-py 0.7.5, AutoROM, python-snappy.
- Downloads the 109 Atari 2600 ROMs via `AutoROM` (license auto-accepted).
- Runs a smoke test that creates a Pong env and prints the obs shape.

Verified eval throughput on M1 (arm64): ~780–1300 steps/sec on CPU depending on algorithm. The same evals on x86_64 Python under Rosetta were 320–700 steps/sec.

After it finishes:

```bash
source .venv/bin/activate
```

### Manual setup (if you skip setup.sh)

```bash
uv venv --python 3.10 .venv
SNAPPY_PREFIX=$(brew --prefix snappy)
CPPFLAGS="-I$SNAPPY_PREFIX/include" LDFLAGS="-L$SNAPPY_PREFIX/lib" \
  VIRTUAL_ENV=.venv uv pip install -r requirements-relaxed.txt
.venv/bin/AutoROM --accept-license
```

> **Why a relaxed requirements file?** The upstream `requirements.txt` pins old versions (torch 2.0.1, mujoco 2.2.2) that fail to install on modern macOS / Python 3.11+. `requirements-relaxed.txt` keeps `gym==0.25.2` and `ale-py==0.7.5` (mandatory — the codebase uses the old gym API and the new ale-py 0.10+ doesn't register old-style env names with old gym), loosens torch / opencv / numpy, drops mujoco (not needed for Atari or any classic-control example here), and adds the `[box2d]` extra so LunarLander and the upstream `gym_env_test` work.

---

## 2. Try a bundled checkpoint

Four pre-trained checkpoints are kept (see §1 for why we removed the fifth). Quickest test:

```bash
# IQN agent on Pong, 2000 eval steps, no tensorboard
python -m deep_rl_zoo.iqn.eval_agent \
    --environment_name=Pong \
    --load_checkpoint_file=./checkpoints/IQN_Pong_2.ckpt \
    --num_iterations=1 \
    --num_eval_steps=2000 \
    --nouse_tensorboard
```

This will:
- Run the agent in greedy (deterministic) mode.
- Print `episode_return` per iteration.
- Record an MP4 of self-play under `recordings/`.

Available bundled checkpoints (and what to expect over 10k steps on M1 arm64):

| File | Algorithm | Game | eval module | Observed `episode_return` (10k steps) |
|---|---|---|---|---|
| `IQN_Pong_2.ckpt` | IQN | Pong | `deep_rl_zoo.iqn.eval_agent` | **−2.50** (loses; undertrained) |
| `PER-DQN_Pong_4.ckpt` | Prioritized DQN | Pong | `deep_rl_zoo.prioritized_dqn.eval_agent` | **+14.0** (wins) |
| `Rainbow_Pong_2.ckpt` | Rainbow | Pong | `deep_rl_zoo.rainbow.eval_agent` | **+10.3** (wins) |
| `PPO-RND_MontezumaRevenge_2.ckpt` | PPO + RND | MontezumaRevenge | `deep_rl_zoo.ppo_rnd.eval_agent` | 0.0 (sparse-reward game; doesn't reach a key in 10k steps) |

Pong is scored from −21 to +21. The **PER-DQN and Rainbow checkpoints win comfortably**; IQN is a partial-training snapshot that loses. The PPO-RND/Montezuma checkpoint is also early-training — Montezuma needs millions of steps to see meaningful exploration.

---

## 3. Train your own

### On Atari

> The upstream defaults are aggressive: `dqn.run_atari` is `num_iterations=100 × num_train_steps=500_000` = **50M frames** per run. On a CPU-only Mac that's days. Always pass smaller numbers for a quick smoke run, then scale up once you know the pipeline works.

```bash
# Quick sanity run (~2 min on M1 CPU): 1 iteration, 5k train steps, 1k eval steps
python -m deep_rl_zoo.dqn.run_atari --environment_name=Pong \
    --num_iterations=1 --num_train_steps=5000 --num_eval_steps=1000 \
    --replay_capacity=10000 --min_replay_size=2000

# Distributed PPO with 8 actors on Breakout (long run; default ≈50M frames)
python -m deep_rl_zoo.ppo.run_atari --environment_name=Breakout --num_actors=8

# Agent57 on a hard exploration game (very long; tune iterations down for a smoke test)
python -m deep_rl_zoo.agent57.run_atari --environment_name=MontezumaRevenge --num_actors=8
```

Each run writes:
- TensorBoard logs to `runs/`
- Checkpoints to `checkpoints/` every N iterations

Watch progress:
```bash
tensorboard --logdir=./runs
```

### On classic control (CartPole, LunarLander) — minutes, useful for sanity checks

```bash
python -m deep_rl_zoo.dqn.run_classic --environment_name=CartPole-v1
python -m deep_rl_zoo.ppo.run_classic --environment_name=LunarLander-v2 --num_actors=4
```

### Algorithms available

Policy-based: `reinforce`, `reinforce_baseline`, `actor_critic`, `a2c`, `sac`, `ppo`, `ppo_icm`, `ppo_rnd`, `impala`
Value-based: `dqn`, `double_dqn`, `prioritized_dqn`, `drqn`, `r2d2`, `ngu`, `agent57`
Distributional: `c51_dqn`, `rainbow`, `qr_dqn`, `iqn`

Each algorithm has the same three entry points: `run_classic`, `run_atari`, `eval_agent`.

---

## 4. Run the upstream test suite

> **Both upstream scripts call `python3` directly — they assume `.venv` is activated.** Without activation, system `python3` runs and you'll get `ModuleNotFoundError: No module named 'absl'`. Activate first: `source .venv/bin/activate`.

There are two upstream test runners and they do different things:

```bash
# Unit tests: 130 tests, ~5 seconds total. Pure-function tests for losses, replay,
# env wrappers, checkpoint serialization, etc. Verified passing on this venv.
./run_unit_tests.sh

# End-to-end tests: ~60 tests, 30-90 seconds each. Actually launches every algorithm's
# run_classic / run_atari / eval_agent for a few hundred steps. Catches whole classes
# of regressions when you start modifying upstream code. Total runtime ~1 hour.
./run_e2e_tests.sh
```

Verified working subsets:

```bash
python -m unit_tests.value_learning_test          # 38 tests, instant
python -m unit_tests.gym_env_test                 # 16 tests (needs gym[box2d])
python -m unit_tests.checkpoint_test              # 8 tests
python -m unit_tests.agent57.run_atari_test       # ~70s — confirms Agent57 actually trains
python -m unit_tests.ppo.run_atari_test           # ~40s — confirms distributed multi-actor training works on this Mac
```

The `checkpoint_test` module reads absl FLAGS during `setUp`, so generic `unittest discover` doesn't work for it — use `python -m unit_tests.checkpoint_test`. The e2e tests write checkpoints under `checkpoints/` and tensorboard logs under `runs/`; both new-file patterns are covered by `.gitignore` so the bundled checkpoints / upstream sample logs stay tracked.

---

## 5. Where to make changes (quick map)

| You want to... | Edit |
|---|---|
| Tweak network architectures | `deep_rl_zoo/networks/{policy,value,curiosity}.py` |
| Add an Atari preprocessing wrapper | `deep_rl_zoo/gym_env.py` |
| Change a loss function | `deep_rl_zoo/{policy_gradient,value_learning,nonlinear_bellman}.py` |
| Modify the experience replay (PER, R2D2 sequence buffer, NGU episodic memory) | `deep_rl_zoo/replay.py` |
| Change distributed actor/learner orchestration | `deep_rl_zoo/main_loop.py`, `deep_rl_zoo/distributed.py` |
| Add a new algorithm | Copy any existing folder (e.g. `deep_rl_zoo/dqn/`) — it has `agent.py`, `run_classic.py`, `run_atari.py`, `eval_agent.py` |
| Tune hyperparameters | Top of each `run_atari.py` (absl-py FLAGS) |

The hyperparameters in the upstream code are **not fine-tuned** (author's own caveat). Any HP sweep is a useful experiment.

---

## 6. External pre-trained Atari agents (different ecosystem)

If you want trained agents on more games right now, the largest public source is the **Stable-Baselines3 Zoo** on HuggingFace: <https://huggingface.co/sb3>. Coverage: DQN, PPO, QR-DQN, A2C × ~25 popular Atari games.

These are **not** integrable with this venv — SB3 needs `gymnasium` + a recent `ale-py`, while deep_rl_zoo is locked to `gym 0.25.2` + `ale-py 0.7.5`. The two stacks are mutually incompatible because ale-py 0.7.5 only auto-registers env names with old gym, and ale-py 0.10+ only auto-registers with gymnasium.

If you want to use them, set up a **separate** venv in a sibling directory (NOT inside this `Atari57/` folder):

```bash
mkdir -p ../Atari57_sb3 && cd ../Atari57_sb3
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install "stable-baselines3[extra]" sb3-contrib huggingface-sb3 "ale-py>=0.10" "gymnasium[atari]"
```

Then load any agent via `huggingface_sb3.load_from_hub(repo_id="sb3/dqn-PongNoFrameskip-v4", filename="dqn-PongNoFrameskip-v4.zip")` and `stable_baselines3.DQN.load(...)`. Note that **this sibling-venv recipe is unverified** — it's the standard SB3 quick-start, but I haven't actually built it on this machine. It exists here as a starting point, not a guarantee.

---

## 7. Project layout

```
Atari57/
├── README_SANDBOX.md          # This file (sandbox usage notes)
├── README.md                  # Upstream README (algorithm catalog + paper refs)
├── QUICK_START.md             # Upstream install notes
├── setup.sh                   # Sandbox setup script
├── requirements-relaxed.txt   # Modern pin set used by setup.sh
├── requirements.txt           # Original upstream pins (kept for reference)
├── run_unit_tests.sh          # Upstream test runner
├── run_e2e_tests.sh           # Upstream end-to-end test runner
├── deep_rl_zoo/               # Upstream source — ~30 algorithms
├── checkpoints/               # 4 bundled .ckpt files + your training output
├── runs/                      # TensorBoard logs (upstream samples + your runs)
├── recordings/                # Self-play MP4s from eval_agent
├── unit_tests/                # Upstream unit tests (129 tests)
├── ideas/                     # Upstream architecture diagrams
└── screenshots/               # Upstream tensorboard screenshots
```

---

## 8. Common gotchas

- **`No module named 'snappy'`** — `python-snappy` needs `brew install snappy` first; `setup.sh` handles this.
- **`Could not initialize NNPACK` (CPU warning)** — harmless on M1.
- **Deprecation warnings about old gym step API and `np.bool8`** — expected. `gym==0.25.2` is pinned because deep_rl_zoo predates the `gym → gymnasium` migration. Don't bump it.
- **`render_mode` warnings during eval** — gym 0.25.2 quirk; the MP4 still renders correctly.
- **Slow training on CPU** — Atari training is GPU-friendly. The upstream code only switches between CUDA and CPU (every `run_*.py` has the same line). On Mac it falls back to CPU even when MPS is available. Patch is one line per `run_*.py` (or use `sed` once):

  ```bash
  # Replace the device-selection line everywhere it appears:
  sed -i '' "s|torch.device('cuda' if torch.cuda.is_available() else 'cpu')|torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))|g" deep_rl_zoo/*/run_classic.py deep_rl_zoo/*/run_atari.py
  ```

  Note: not every op in the codebase has an MPS kernel in older torch builds — if you hit `NotImplementedError: ... aten::xxx` falling back to CPU on M1, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to route unsupported ops to CPU automatically.
- **Architecture mismatches when loading older checkpoints** — upstream commit `cd860e8` (June 2023) refactored conv-net heads from `Linear` to `Sequential(Linear, ReLU, Linear)`. Any pre-2023 .ckpt for the affected algorithms (PPO, A2C, etc.) will fail with `Missing key(s) in state_dict: policy_head.0.weight, ...`. There's no clean back-compat path; just retrain.

---

## 9. Bibliography

Canonical reads for the headline algorithms in this repo:

- **Agent57** (Badia et al., DeepMind, 2020) — first agent above human on all 57 Atari games. <https://arxiv.org/abs/2003.13350>
- **R2D2** (Kapturowski et al., DeepMind, 2019) — distributed recurrent replay foundation. <https://openreview.net/pdf?id=r1lyTjAqYX>
- **NGU** (Badia et al., 2020) — exploration curriculum that Agent57 builds on. <https://arxiv.org/abs/2002.06038>
- **IQN** (Dabney et al., 2018) — implicit quantile distributional RL. <https://arxiv.org/abs/1806.06923>
- **Rainbow** (Hessel et al., 2017) — combined DQN improvements. <https://arxiv.org/abs/1710.02298>

These map to `deep_rl_zoo/{agent57,r2d2,ngu,iqn,rainbow}/`.
