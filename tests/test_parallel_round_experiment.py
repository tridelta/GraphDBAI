from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from experience_graph.scripts.run_parallel_round_experiment import main as parallel_main


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "world_cases" / "textcraft_cases.yaml"
RULES = ROOT / "world_cases" / "textcraft_rules.yaml"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_parallel_round_runner_integrates_after_each_round(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parallel_round_experiment",
            "--agent",
            "scripted",
            "--cases",
            str(CASES),
            "--rules",
            str(RULES),
            "--case-ids",
            "TC_CRAFT_001,TC_CRAFT_002",
            "--task-id",
            "diamond_set",
            "--rounds",
            "2",
            "--max-workers",
            "2",
            "--max-steps",
            "10",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "parallel_test",
        ],
    )

    parallel_main()

    run_path = run_dir / "parallel_test"
    config = yaml.safe_load((run_path / "config.yaml").read_text(encoding="utf-8"))
    assert config["parallel_protocol"] == "round_batch"
    assert config["round_case_count"] == 2

    metrics = read_jsonl(run_path / "metrics.jsonl")
    assert len(metrics) == 4
    assert [(row["round_index"], row["case_order_index"]) for row in metrics] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all(row["parallel_protocol"] == "round_batch" for row in metrics)
    assert [row["episode_index"] for row in metrics] == [0, 1, 2, 3]
    assert metrics[-1]["graph_nodes"] > 0

    steps = read_jsonl(run_path / "steps.jsonl")
    assert steps
    assert all("round_index" in row and "case_order_index" in row for row in steps)
    assert (run_path / "graph_nodes.jsonl").exists()
    assert (run_path / ".parallel_tmp" / "round_000" / "case_000_TC_CRAFT_001" / "steps.jsonl").exists()


def test_parallel_round_runner_resumes_from_completed_round(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs"
    base_argv = [
        "run_parallel_round_experiment",
        "--agent",
        "scripted",
        "--cases",
        str(CASES),
        "--rules",
        str(RULES),
        "--case-ids",
        "TC_CRAFT_001,TC_CRAFT_002",
        "--task-id",
        "diamond_set",
        "--rounds",
        "2",
        "--max-workers",
        "2",
        "--max-steps",
        "10",
        "--max-budget-rmb",
        "0.0",
        "--run-dir",
        str(run_dir),
        "--run-id",
        "parallel_resume_test",
    ]
    monkeypatch.setattr(sys, "argv", base_argv)
    parallel_main()

    run_path = run_dir / "parallel_resume_test"
    assert len(read_jsonl(run_path / "metrics.jsonl")) == 2

    monkeypatch.setattr(sys, "argv", [*base_argv, "--resume"])
    parallel_main()

    metrics = read_jsonl(run_path / "metrics.jsonl")
    assert len(metrics) == 4
    assert [(row["round_index"], row["case_order_index"]) for row in metrics] == [(0, 0), (0, 1), (1, 0), (1, 1)]
