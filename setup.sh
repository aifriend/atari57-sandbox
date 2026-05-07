#!/usr/bin/env bash
# One-shot setup for the Atari57 sandbox on macOS.
# Idempotent — safe to re-run.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> Atari57 setup (root: $ROOT)"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

# 1. System deps via Homebrew. snappy is required by deep_rl_zoo.replay (transition
#    compression); ffmpeg is required for self-play MP4 recording during eval.
if [[ "$(uname -s)" == "Darwin" ]]; then
  if command -v brew >/dev/null 2>&1; then
    for pkg in snappy ffmpeg; do
      if ! brew list --formula | grep -qx "$pkg"; then
        echo "==> brew install $pkg"
        brew install "$pkg"
      fi
    done
  else
    echo "WARN: Homebrew not found. Install snappy and ffmpeg manually before continuing."
  fi
fi

# 2. Python venv. On Apple Silicon explicitly request the aarch64 build: the
#    Homebrew /usr/local/opt/python@3.10 is x86_64 and would run under Rosetta
#    (~2x slower for PyTorch). uv downloads a managed arm64 cpython if needed.
if [[ ! -d ".venv" ]]; then
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "==> Creating .venv with native arm64 Python 3.10"
    uv venv --python cpython-3.10-macos-aarch64-none .venv
  else
    echo "==> Creating .venv with Python 3.10"
    uv venv --python 3.10 .venv
  fi
fi

# 3. Python deps. python-snappy needs CPPFLAGS/LDFLAGS pointing at the snappy headers.
#    On Mac that's brew's snappy; on Linux the default include/lib paths usually work
#    if libsnappy-dev is installed (apt-get install libsnappy-dev).
echo "==> Installing Python dependencies"
INSTALL_ENV=()
if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
  SNAPPY_PREFIX="$(brew --prefix snappy 2>/dev/null || true)"
  if [[ -n "$SNAPPY_PREFIX" && -d "$SNAPPY_PREFIX/include" ]]; then
    INSTALL_ENV=(CPPFLAGS="-I${SNAPPY_PREFIX}/include" LDFLAGS="-L${SNAPPY_PREFIX}/lib")
  else
    echo "WARN: brew snappy not found; python-snappy install may fail. Run: brew install snappy"
  fi
elif [[ "$(uname -s)" == "Linux" ]]; then
  if ! ldconfig -p 2>/dev/null | grep -q libsnappy; then
    echo "WARN: libsnappy not found; python-snappy install may fail. Run: sudo apt-get install libsnappy-dev"
  fi
fi
env "${INSTALL_ENV[@]}" VIRTUAL_ENV=.venv uv pip install -r requirements-relaxed.txt

# 4. Atari ROMs.
if [[ ! -f .venv/lib/python3.10/site-packages/AutoROM/roms/pong.bin ]]; then
  echo "==> Installing Atari ROMs via AutoROM"
  .venv/bin/AutoROM --accept-license
else
  echo "==> Atari ROMs already installed"
fi

# 5. Smoke test (warnings suppressed — deep_rl_zoo's gym 0.25.2 is intentionally pinned).
echo "==> Smoke test"
.venv/bin/python -W ignore -c "
import warnings, gym.logger
gym.logger.set_level(40)  # ERROR
warnings.filterwarnings('ignore')
from deep_rl_zoo import gym_env
env = gym_env.create_atari_environment(env_name='Pong')
obs = env.reset()
assert obs.shape == (4, 84, 84), f'unexpected obs shape {obs.shape}'
print(f'   OK - obs shape {obs.shape}, action_space {env.action_space.n}')
"

cat <<'EOF'

==> Done. Activate with:
    source .venv/bin/activate

==> Quick eval against a bundled checkpoint:
    python -m deep_rl_zoo.iqn.eval_agent \
        --environment_name=Pong \
        --load_checkpoint_file=./checkpoints/IQN_Pong_2.ckpt \
        --num_iterations=1 --num_eval_steps=2000 --nouse_tensorboard
EOF
