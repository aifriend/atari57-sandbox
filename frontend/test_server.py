"""Tests for the FastAPI sidecar.

Run from the repo root with:
    PYTHONPATH=. .venv/bin/pytest frontend/test_server.py -v

Tests rely on the bundled checkpoints (4 .ckpt files in checkpoints/) and
the upstream tensorboard sample logs in runs/. They don't spawn real
training subprocesses — those code paths only get a happy-path check
that they validate inputs correctly.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from frontend.server import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["checkpoints_dir_exists"] is True
    assert body["runs_dir_exists"] is True


def test_list_checkpoints_has_four_bundled() -> None:
    r = client.get("/api/checkpoints")
    body = r.json()
    assert r.status_code == 200
    # The 4 bundled .ckpt files survive the round-trip parse.
    filenames = {item["filename"] for item in body["items"]}
    assert "IQN_Pong_2.ckpt" in filenames
    assert "PER-DQN_Pong_4.ckpt" in filenames
    assert "Rainbow_Pong_2.ckpt" in filenames
    assert "PPO-RND_MontezumaRevenge_2.ckpt" in filenames

    # Each item carries parsed metadata.
    for item in body["items"]:
        assert "algo_module" in item
        assert "algo_label" in item
        assert "game" in item
        assert "size_bytes" in item
        assert item["size_bytes"] > 0


def test_filename_to_module_mapping() -> None:
    r = client.get("/api/checkpoints").json()
    by_fn = {i["filename"]: i for i in r["items"]}
    assert by_fn["IQN_Pong_2.ckpt"]["algo_module"] == "iqn"
    assert by_fn["PER-DQN_Pong_4.ckpt"]["algo_module"] == "prioritized_dqn"
    assert by_fn["Rainbow_Pong_2.ckpt"]["algo_module"] == "rainbow"
    assert by_fn["PPO-RND_MontezumaRevenge_2.ckpt"]["algo_module"] == "ppo_rnd"


def test_list_games_has_57_with_checkpoint_flag() -> None:
    r = client.get("/api/games").json()
    assert r["count"] == 57
    by_name = {g["name"]: g for g in r["items"]}
    assert by_name["Pong"]["has_checkpoint"] is True
    assert by_name["MontezumaRevenge"]["has_checkpoint"] is True
    assert by_name["Breakout"]["has_checkpoint"] is False  # no bundled Breakout ckpt


def test_list_algorithms_has_20_grouped() -> None:
    r = client.get("/api/algorithms").json()
    assert r["count"] == 20
    families = {item["family"] for item in r["items"]}
    assert families == {"value", "distributional", "policy"}
    # Algos with bundled checkpoints have non-zero counts.
    by_module = {a["module"]: a for a in r["items"]}
    assert by_module["iqn"]["checkpoint_count"] >= 1
    assert by_module["rainbow"]["checkpoint_count"] >= 1
    # Unbundled ones don't.
    assert by_module["dqn"]["checkpoint_count"] == 0


def test_list_runs_returns_event_files() -> None:
    r = client.get("/api/runs").json()
    assert r["count"] >= 1
    # Pong-Rainbow-train is one of the upstream-bundled samples.
    names = {item["name"] for item in r["items"]}
    assert any("Rainbow" in n for n in names)


def test_run_scalars_returns_episode_return_for_known_run() -> None:
    r = client.get(
        "/api/runs/PongNoFrameskip-v4-Rainbow-train/scalars",
        params={
            "tag": "performance(env_steps)/episode_return",
            "max_points": 10,
        },
    )
    assert r.status_code == 200
    body = r.json()
    tag = "performance(env_steps)/episode_return"
    assert tag in body["tags"]
    points = body["tags"][tag]
    assert len(points) >= 1
    # Each point has a step + value.
    for p in points:
        assert isinstance(p["step"], int)
        assert isinstance(p["value"], float)
    # Pong returns are within [-21, 21] — nothing crazy.
    assert all(-21.5 <= p["value"] <= 21.5 for p in points)


def test_run_scalars_rejects_path_traversal() -> None:
    r = client.get("/api/runs/..%2F..%2Fetc/scalars")
    body = r.json()
    # Either a 404 or an error field is acceptable; the contract is "no leak".
    assert "error" in body or r.status_code in (400, 404)


def test_run_scalars_unknown_run_returns_error() -> None:
    body = client.get("/api/runs/this-run-does-not-exist/scalars").json()
    assert body.get("error") == "run not found"


def test_recordings_endpoint_lists_or_empty() -> None:
    r = client.get("/api/recordings").json()
    assert "count" in r and "items" in r
    for item in r["items"]:
        assert item["filename"].endswith(".mp4")
        assert item["url"].startswith("/api/recordings/")


def test_recordings_404_on_missing_file() -> None:
    r = client.get("/api/recordings/does/not/exist.mp4")
    assert r.status_code == 404


def test_recordings_rejects_path_traversal() -> None:
    r = client.get("/api/recordings/..%2F..%2Fetc%2Fpasswd")
    assert r.status_code in (400, 404)


def test_training_start_requires_algo_and_game() -> None:
    r = client.post("/api/training/start", json={})
    assert r.status_code == 200
    assert "error" in r.json()


def test_training_start_rejects_unknown_algo() -> None:
    r = client.post("/api/training/start", json={"algo": "not_an_algo", "game": "Pong"})
    body = r.json()
    assert "error" in body
    assert "unknown" in body["error"]


def test_training_jobs_list_works() -> None:
    r = client.get("/api/training/jobs")
    assert r.status_code == 200
    assert "items" in r.json()


def test_comparison_validates_input() -> None:
    r = client.post("/api/comparison/run", json={})
    body = r.json()
    assert "error" in body


def test_static_index_served_at_root() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert b"ATARI57" in r.content
    assert b'<script src="app.js"' in r.content


def test_app_js_served() -> None:
    r = client.get("/app.js")
    assert r.status_code == 200
    assert b"loadCatalog" in r.content


# ───────── hardening: training start / comparison / CORS ─────────


def _train(**overrides):
    body = {"algo": "dqn", "game": "Pong"}
    body.update(overrides)
    return client.post("/api/training/start", json=body).json()


def test_training_start_rejects_unknown_game() -> None:
    assert "unknown game" in _train(game="NotAGame")["error"]


def test_training_start_rejects_game_with_injection() -> None:
    assert "error" in _train(game="Pong --checkpoint_dir=/tmp/x")


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_train_steps", "abc"),
        ("num_train_steps", -5),
        ("num_train_steps", 1.5),
        ("num_iterations", 0),
        ("num_eval_steps", 10**12),
        ("num_train_steps", True),
        ("num_train_steps", [1]),
    ],
)
def test_training_start_rejects_bad_numbers(field: str, value) -> None:
    assert "error" in _train(**{field: value})


@pytest.mark.parametrize(
    "extra",
    [
        ["--checkpoint_dir=/tmp/evil"],
        ["--results_csv_path=/etc/passwd"],
        ["--tag=x"],
        ["--learning_rate=abc"],
        ["--learning_rate=nan"],
        ["--learning_rate=5"],
        ["--batch_size=0"],
        ["--batch_size=1.5"],
        ["-m", "os"],
        ["learning_rate=0.1"],
        [123],
        "--learning_rate=0.1",
    ],
)
def test_training_start_rejects_disallowed_extra_args(extra) -> None:
    assert "error" in _train(extra_args=extra)


def test_extra_args_allowlist_accepts_and_normalises() -> None:
    from frontend.server import _validate_extra_args

    assert _validate_extra_args(None) == []
    assert _validate_extra_args(["--learning_rate=0.0005", "--batch_size=64", "--nouse_tensorboard", "--clip_grad"]) == [
        "--learning_rate=0.0005",
        "--batch_size=64",
        "--nouse_tensorboard",
        "--clip_grad",
    ]


def test_training_start_validation_happens_before_spawn(monkeypatch) -> None:
    import frontend.server as server

    async def boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("subprocess spawned despite invalid input")

    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", boom)
    assert "error" in _train(extra_args=["--checkpoint_dir=/tmp/x"])


def test_resolve_checkpoint_accepts_bundled_and_rejects_traversal() -> None:
    from frontend.server import CHECKPOINTS_DIR, _resolve_checkpoint

    ok = _resolve_checkpoint("IQN_Pong_2.ckpt")
    assert ok is not None and ok.parent == CHECKPOINTS_DIR.resolve()
    for bad in ["../IQN_Pong_2.ckpt", "/etc/passwd", "sub/IQN_Pong_2.ckpt", "IQN_Pong_2.txt", "", None, 5, "Nope_Pong_9.ckpt"]:
        assert _resolve_checkpoint(bad) is None


def test_comparison_rejects_traversal_and_bad_steps() -> None:
    body = client.post("/api/comparison/run", json={"checkpoints": ["../README.md"], "num_steps": 10}).json()
    assert body["results"][0]["error"] == "not found"
    assert "error" in client.post("/api/comparison/run", json={"checkpoints": ["IQN_Pong_2.ckpt"], "num_steps": "x"}).json()
    assert "error" in client.post("/api/comparison/run", json={"checkpoints": "IQN_Pong_2.ckpt"}).json()


def test_eval_stream_validates_checkpoint_and_numbers() -> None:
    with client.websocket_connect("/api/eval/stream") as ws:
        ws.send_json({"action": "start", "checkpoint": "../README.md"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"action": "start", "checkpoint": "IQN_Pong_2.ckpt", "num_steps": -1})
        assert ws.receive_json()["type"] == "error"


def test_cors_allows_loopback_origin_only() -> None:
    ok = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers
    assert bad.status_code == 200
