"""Tests for the FastAPI sidecar.

Run from the repo root with:
    PYTHONPATH=. .venv/bin/pytest frontend/test_server.py -v

Tests rely on the bundled checkpoints (4 .ckpt files in checkpoints/) and
the upstream tensorboard sample logs in runs/. They don't spawn real
training subprocesses — those code paths only get a happy-path check
that they validate inputs correctly.
"""
from __future__ import annotations

import json

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
