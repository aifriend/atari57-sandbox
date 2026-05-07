"""Streaming eval subprocess: runs a deep_rl_zoo greedy agent and writes
JSON-per-line to stdout.

Spawned by the FastAPI WebSocket endpoint `/api/eval/stream`. Decoupled into
its own process so:
  1. absl FLAGS / module-level imports don't pollute the API server.
  2. Killing the subprocess on client disconnect is clean.
  3. Each eval is fully isolated.

Output format (one JSON object per line):
  {"type": "init",    "game": "Pong", "algo": "iqn", "actions": ["NOOP", "FIRE", ...]}
  {"type": "step",    "step": 1, "frame_b64": "...", "reward": 0.0, "action": 2,
                      "ep_step": 1, "ep_return": 0.0}
  {"type": "episode", "episode_return": -2.0, "episode_steps": 1820}
  {"type": "done",    "total_steps": 5000}
  {"type": "error",   "message": "..."}

Usage:
  python -m frontend.stream_eval \
      --checkpoint=checkpoints/IQN_Pong_2.ckpt \
      --algo=iqn \
      --game=Pong \
      --num-steps=2000 \
      --frame-stride=2

Currently supports the four bundled checkpoints' algorithms:
  iqn, prioritized_dqn (DQN-style), rainbow, ppo_rnd.
Easy to extend — add an entry in `ALGO_FACTORIES`.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

import gym
import gym.logger

gym.logger.set_level(40)  # suppress gym deprecation spam

from deep_rl_zoo import gym_env, greedy_actors  # noqa: E402
from deep_rl_zoo.checkpoint import PyTorchCheckpoint  # noqa: E402
from deep_rl_zoo.networks.value import (  # noqa: E402
    DqnConvNet,
    IqnConvNet,
    RainbowDqnConvNet,
)
from deep_rl_zoo.networks.policy import RndActorCriticConvNet  # noqa: E402


class TimeStep(NamedTuple):
    """Mirrors deep_rl_zoo.types.TimeStep — duplicated to avoid coupling."""

    observation: np.ndarray | None
    reward: float | None
    done: bool | None
    first: bool | None


def _emit(obj: dict) -> None:
    """Write one JSON object + newline to stdout, flush immediately."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _png_b64(frame: np.ndarray) -> str:
    """Encode a (H, W, 3) uint8 RGB frame as base64 PNG."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG", optimize=False, compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ───────── per-algo construction ─────────


def _build_iqn(state_dim, action_dim, device, checkpoint_path, env_name) -> tuple:
    network = IqnConvNet(state_dim=state_dim, action_dim=action_dim, latent_dim=64)
    ckpt = PyTorchCheckpoint(environment_name=env_name, agent_name="IQN", restore_only=True)
    ckpt.register_pair(("network", network))
    ckpt.restore(checkpoint_path)
    network.eval()
    actor = greedy_actors.IqnEpsilonGreedyActor(
        network=network,
        exploration_epsilon=0.01,
        random_state=np.random.RandomState(0),
        device=device,
        tau_samples=64,
    )
    return network, actor


def _build_prioritized_dqn(state_dim, action_dim, device, checkpoint_path, env_name) -> tuple:
    network = DqnConvNet(state_dim=state_dim, action_dim=action_dim)
    ckpt = PyTorchCheckpoint(environment_name=env_name, agent_name="DQN", restore_only=True)
    ckpt.register_pair(("network", network))
    ckpt.restore(checkpoint_path)
    network.eval()
    actor = greedy_actors.EpsilonGreedyActor(
        network=network,
        exploration_epsilon=0.01,
        random_state=np.random.RandomState(0),
        device=device,
    )
    return network, actor


def _build_rainbow(state_dim, action_dim, device, checkpoint_path, env_name) -> tuple:
    atoms = torch.linspace(-10.0, 10.0, 51)
    network = RainbowDqnConvNet(state_dim=state_dim, action_dim=action_dim, atoms=atoms)
    ckpt = PyTorchCheckpoint(environment_name=env_name, agent_name="Rainbow", restore_only=True)
    ckpt.register_pair(("network", network))
    ckpt.restore(checkpoint_path)
    network.eval()
    actor = greedy_actors.EpsilonGreedyActor(
        network=network,
        exploration_epsilon=0.01,
        random_state=np.random.RandomState(0),
        device=device,
    )
    return network, actor


def _build_ppo_rnd(state_dim, action_dim, device, checkpoint_path, env_name) -> tuple:
    network = RndActorCriticConvNet(state_dim=state_dim, action_dim=action_dim)
    ckpt = PyTorchCheckpoint(environment_name=env_name, agent_name="PPO-RND", restore_only=True)
    ckpt.register_pair(("policy_network", network))
    ckpt.restore(checkpoint_path)
    network.eval()
    actor = greedy_actors.PolicyGreedyActor(
        network=network,
        device=device,
        name="PPO-RND-greedy",
    )
    return network, actor


ALGO_FACTORIES: dict[str, Callable] = {
    "iqn": _build_iqn,
    "prioritized_dqn": _build_prioritized_dqn,
    "rainbow": _build_rainbow,
    "ppo_rnd": _build_ppo_rnd,
}


# Standard ALE/Atari action set names (action_dim varies by game; we report
# the first N actions matching env.action_space.n).
ATARI_ACTION_NAMES = [
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
    "UP-RIGHT", "UP-LEFT", "DOWN-RIGHT", "DOWN-LEFT",
    "UP-FIRE", "RIGHT-FIRE", "LEFT-FIRE", "DOWN-FIRE",
    "UP-RIGHT-FIRE", "UP-LEFT-FIRE", "DOWN-RIGHT-FIRE", "DOWN-LEFT-FIRE",
]


# ───────── main loop ─────────


def run(checkpoint: Path, algo: str, game: str, num_steps: int, frame_stride: int) -> int:
    if algo not in ALGO_FACTORIES:
        _emit({"type": "error", "message": f"unsupported algo {algo!r}; supported: {list(ALGO_FACTORIES)}"})
        return 2
    if not checkpoint.exists():
        _emit({"type": "error", "message": f"checkpoint not found: {checkpoint}"})
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym_env.create_atari_environment(
        env_name=game,
        frame_height=84,
        frame_width=84,
        frame_skip=4,
        frame_stack=4,
        max_episode_steps=58000,
        seed=1,
        noop_max=30,
        terminal_on_life_loss=False,
        clip_reward=False,
    )
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n

    _build = ALGO_FACTORIES[algo]
    network, actor = _build(state_dim, action_dim, device, str(checkpoint), game)

    actions = ATARI_ACTION_NAMES[:action_dim]
    _emit({"type": "init", "game": game, "algo": algo, "actions": actions, "action_dim": action_dim})

    obs = env.reset()
    timestep = TimeStep(observation=obs, reward=0.0, done=False, first=True)
    actor.reset()

    ep_step = 0
    ep_return = 0.0
    total_steps = 0

    while total_steps < num_steps:
        action = actor.step(timestep)
        try:
            action_int = int(action)
        except (TypeError, ValueError):
            action_int = int(np.asarray(action).item())

        next_obs, reward, done, _info = env.step(action_int)
        ep_step += 1
        ep_return += float(reward)
        total_steps += 1

        if total_steps % frame_stride == 0 or done:
            try:
                raw = env.unwrapped.render(mode="rgb_array")
            except TypeError:
                raw = env.unwrapped.render()
            if raw is None:
                _emit({"type": "error", "message": "env.render returned None — try render_mode='rgb_array' env"})
                return 3
            _emit(
                {
                    "type": "step",
                    "step": total_steps,
                    "ep_step": ep_step,
                    "frame_b64": _png_b64(raw),
                    "reward": float(reward),
                    "action": action_int,
                    "ep_return": ep_return,
                }
            )

        if done:
            _emit({"type": "episode", "episode_return": ep_return, "episode_steps": ep_step})
            obs = env.reset()
            actor.reset()
            timestep = TimeStep(observation=obs, reward=0.0, done=False, first=True)
            ep_step = 0
            ep_return = 0.0
        else:
            obs = next_obs
            timestep = TimeStep(observation=obs, reward=float(reward), done=False, first=False)

    _emit({"type": "done", "total_steps": total_steps})
    env.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--algo", required=True, choices=sorted(ALGO_FACTORIES))
    parser.add_argument("--game", required=True)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--frame-stride", type=int, default=2,
                        help="Emit a frame every N env steps (frame_skip=4 already applied)")
    args = parser.parse_args()

    try:
        return run(args.checkpoint, args.algo, args.game, args.num_steps, args.frame_stride)
    except Exception as exc:  # noqa: BLE001
        _emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
