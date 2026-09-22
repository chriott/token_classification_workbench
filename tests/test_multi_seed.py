from __future__ import annotations

import json
import math
from pathlib import Path

from token_classification.config import TrainingConfig
from token_classification.multi_seed import aggregate_nervaluate_results, run_multi_seed_final_training
from token_classification.utils import write_json


def make_nervaluate_result(seed: int):
    offset = (seed - 10) / 10
    return {
        "labels": ["EMPTY", "PERSON"],
        "rollups": {
            "partial": {
                "overall": {
                    "micro": {"precision": 0.7 + offset, "recall": 0.75 + offset, "f1": 0.8 + offset},
                    "macro": {"precision": 0.6 + offset, "recall": 0.65 + offset, "f1": 0.7 + offset},
                }
            }
        },
        "per_tag_results": {
            "EMPTY": {
                "gold_support": 0,
                "evaluable": False,
                "scenarios": {"partial": None},
            },
            "PERSON": {
                "gold_support": 4,
                "evaluable": True,
                "scenarios": {
                    "partial": {
                        "precision": 0.75 + offset,
                        "recall": 0.8 + offset,
                        "f1": 0.85 + offset,
                    }
                },
            },
        },
    }


def test_aggregate_nervaluate_results_calculates_sample_standard_deviation():
    summary = aggregate_nervaluate_results({10: make_nervaluate_result(10), 12: make_nervaluate_result(12)})

    micro_f1 = summary["scenarios"]["partial"]["overall"]["micro"]["f1"]
    person_f1 = summary["scenarios"]["partial"]["labels"]["PERSON"]["metrics"]["f1"]
    empty = summary["scenarios"]["partial"]["labels"]["EMPTY"]

    assert micro_f1["mean"] == 0.9
    assert math.isclose(micro_f1["std"], math.sqrt(0.02))
    assert micro_f1["values_by_seed"] == {"10": 0.8, "12": 1.0}
    assert person_f1["mean"] == 0.95
    assert empty == {"gold_support": 0, "evaluable": False, "metrics": {}}


def test_run_multi_seed_final_training_isolates_runs_and_writes_aggregate(tmp_path):
    config = TrainingConfig(output_dir=str(tmp_path), run_name="experiment")
    seen_configs = []
    save_model_values = []

    def fake_pipeline_runner(seed_config, *, final_training_mode, save_model):
        assert final_training_mode is True
        seen_configs.append(seed_config)
        save_model_values.append(save_model)
        run_dir = Path(seed_config.output_dir) / seed_config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "model.safetensors").write_text("weights", encoding="utf-8")
        write_json(
            run_dir / "nervaluate" / "nervaluate_test.json",
            make_nervaluate_result(seed_config.split_seed),
        )
        return run_dir

    experiment_dir = run_multi_seed_final_training(
        config,
        [10, 12],
        pipeline_runner=fake_pipeline_runner,
        timestamp="2026-09-02_10-30-25",
    )

    assert experiment_dir == tmp_path / "experiment_2026-09-02_10-30-25"
    assert [item.split_seed for item in seen_configs] == [10, 12]
    assert [item.run_name for item in seen_configs] == ["seed_10", "seed_12"]
    assert save_model_values == [True, False]
    assert (experiment_dir / "seed_10" / "model.safetensors").exists()
    assert not (experiment_dir / "seed_12" / "model.safetensors").exists()
    aggregate_dir = experiment_dir / "aggregate"
    assert (aggregate_dir / "nervaluate_multi_seed.json").is_file()
    assert (aggregate_dir / "nervaluate_multi_seed.csv").is_file()
    assert (aggregate_dir / "nervaluate_multi_seed.txt").is_file()
    manifest = json.loads((aggregate_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["seeds"] == [10, 12]
    assert manifest["retained_model_seed"] == 10


def test_run_multi_seed_final_training_rejects_duplicate_seeds(tmp_path):
    config = TrainingConfig(output_dir=str(tmp_path), run_name="experiment")

    try:
        run_multi_seed_final_training(config, [10, 10], pipeline_runner=lambda *_args, **_kwargs: None)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("Expected duplicate seeds to be rejected.")


def test_run_multi_seed_final_training_records_failure(tmp_path):
    config = TrainingConfig(output_dir=str(tmp_path), run_name="experiment")

    def failing_runner(*_args, **_kwargs):
        raise RuntimeError("training failed")

    try:
        run_multi_seed_final_training(
            config,
            [10],
            pipeline_runner=failing_runner,
            timestamp="2026-09-02_10-30-25",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected the training failure to be propagated.")

    manifest_path = tmp_path / "experiment_2026-09-02_10-30-25" / "aggregate" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"] == "RuntimeError: training failed"
