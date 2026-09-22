import json
import random
from pathlib import Path

from token_classification.config import CrossValidationConfig, SweepConfig, SweepParameter, TrainingConfig
from token_classification.cross_validation import FoldSpec
from token_classification.sweep import enumerate_grid_trial_overrides, run_sweep, sample_parameter_value, sample_trial_overrides
from token_classification.utils import write_json


def test_sample_parameter_value_from_values():
    parameter = SweepParameter(values=[8, 16], type="int")
    rng = random.Random(7)

    value = sample_parameter_value(parameter, rng)

    assert value in {8, 16}
    assert isinstance(value, int)


def test_sample_parameter_value_from_log_float_range():
    parameter = SweepParameter(min=1.0e-5, max=1.0e-4, type="float", log=True)
    rng = random.Random(11)

    value = sample_parameter_value(parameter, rng)

    assert 1.0e-5 <= value <= 1.0e-4
    assert isinstance(value, float)


def test_sample_trial_overrides_uses_defined_search_space():
    sweep_config = SweepConfig(
        name="demo",
        output_dir="outputs/sweeps",
        num_trials=2,
        search_strategy="random",
        seed=25,
        objective_metric="eval_f1_macro",
        objective_mode="max",
        base_config=TrainingConfig(
            train_file="data/train.csv",
            validation_file="data/validation.csv",
            test_file="data/test.csv",
        ),
        search_space={
            "train_batch_size": SweepParameter(values=[8, 16], type="int"),
            "train_weight_decay": SweepParameter(values=[0.0, 0.01], type="float"),
        },
    )
    rng = random.Random(3)

    overrides = sample_trial_overrides(sweep_config, rng)

    assert set(overrides.keys()) == {"train_batch_size", "train_weight_decay"}
    assert overrides["train_batch_size"] in {8, 16}
    assert overrides["train_weight_decay"] in {0.0, 0.01}


def test_sweep_config_rejects_scalar_search_space_values():
    raw_config = {
        "name": "demo",
        "base_config": {
            "train_file": "data/train.csv",
            "validation_file": "data/validation.csv",
            "test_file": "data/test.csv",
        },
        "search_space": {
            "train_batch_size": "16",
        },
    }

    try:
        SweepConfig.from_mapping(raw_config)
    except ValueError as exc:
        assert "must be defined as a mapping or list" in str(exc)
    else:
        raise AssertionError("Expected malformed scalar sweep parameter to raise ValueError.")


def test_grid_search_enumerates_cartesian_product():
    sweep_config = SweepConfig(
        name="demo_grid",
        output_dir="outputs/sweeps",
        num_trials=10,
        search_strategy="grid",
        seed=25,
        objective_metric="eval_f1_macro",
        objective_mode="max",
        base_config=TrainingConfig(
            train_file="data/train.csv",
            validation_file="data/validation.csv",
            test_file="data/test.csv",
        ),
        search_space={
            "train_batch_size": SweepParameter(values=[2, 4], type="int"),
            "gradient_accumulation_steps": SweepParameter(values=[1, 2], type="int"),
        },
    )

    overrides = enumerate_grid_trial_overrides(sweep_config)

    assert overrides == [
        {"train_batch_size": 2, "gradient_accumulation_steps": 1},
        {"train_batch_size": 2, "gradient_accumulation_steps": 2},
        {"train_batch_size": 4, "gradient_accumulation_steps": 1},
        {"train_batch_size": 4, "gradient_accumulation_steps": 2},
    ]


def test_grid_search_requires_explicit_values():
    sweep_config = SweepConfig(
        name="demo_grid_invalid",
        output_dir="outputs/sweeps",
        num_trials=4,
        search_strategy="grid",
        seed=25,
        objective_metric="eval_f1_macro",
        objective_mode="max",
        base_config=TrainingConfig(
            train_file="data/train.csv",
            validation_file="data/validation.csv",
            test_file="data/test.csv",
        ),
        search_space={
            "train_learning_rate": SweepParameter(min=1.0e-5, max=8.0e-5, type="float", log=True),
        },
    )

    try:
        sweep_config.validate()
    except ValueError as exc:
        assert "must define explicit 'values' for grid search" in str(exc)
    else:
        raise AssertionError("Expected grid search with ranged parameter to raise ValueError.")


def test_run_sweep_persists_incremental_leaderboard_and_summary(tmp_path, monkeypatch):
    sweep_config = SweepConfig(
        name="demo_incremental",
        output_dir=str(tmp_path / "outputs"),
        num_trials=2,
        search_strategy="grid",
        seed=25,
        objective_metric="eval_f1_macro",
        objective_mode="max",
        base_config=TrainingConfig(
            train_file="data/train.csv",
            validation_file="data/validation.csv",
            test_file="data/test.csv",
        ),
        search_space={
            "train_batch_size": SweepParameter(values=[2, 4], type="int"),
        },
    )

    state = {"calls": 0, "save_model_values": []}

    def fake_run_pipeline(config, run_test_evaluation=False, save_model=True):
        state["calls"] += 1
        state["save_model_values"].append(save_model)
        run_dir = Path(config.output_dir) / config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = run_dir / f"checkpoint-{state['calls']}"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors").write_text("large model placeholder", encoding="utf-8")
        (run_dir / "run-metadata.txt").write_text("retain me", encoding="utf-8")
        if state["calls"] == 2:
            raise RuntimeError("boom")
        write_json(
            run_dir / "run_summary.json",
            {"validation_metrics": {"eval_f1_macro": 0.7}},
        )
        return run_dir

    monkeypatch.setattr("token_classification.sweep.run_pipeline", fake_run_pipeline)

    sweep_root = run_sweep(sweep_config, timestamp="2026-09-02_10-30-25")
    leaderboard_path = sweep_root / "leaderboard.csv"
    summary_path = sweep_root / "summary.json"

    assert leaderboard_path.exists()
    assert summary_path.exists()
    assert sweep_root.name == "demo_incremental_2026-09-02_10-30-25"

    leaderboard_rows = leaderboard_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(leaderboard_rows) == 3
    assert "trial_001" in leaderboard_rows[1]
    assert "trial_002" in leaderboard_rows[2]
    assert "failed" in leaderboard_rows[2]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["completed_trials"] == 1
    assert summary["failed_trials"] == 1
    assert summary["best_trial"]["run_name"] == "trial_001"
    assert state["save_model_values"] == [False, False]
    assert not list(sweep_root.glob("trial_*/checkpoint-*"))
    assert (sweep_root / "trial_001" / "run-metadata.txt").exists()
    assert (sweep_root / "trial_002" / "run-metadata.txt").exists()


def test_run_sweep_aggregates_fixed_cross_validation_folds_without_models(tmp_path, monkeypatch):
    sweep_config = SweepConfig(
        name="cv_demo",
        output_dir=str(tmp_path / "outputs"),
        num_trials=1,
        search_strategy="grid",
        seed=137,
        objective_metric="eval_nervaluate_partial_micro_f1",
        objective_mode="max",
        cross_validation=CrossValidationConfig(folds=2, seed=211, group_column="document_id"),
        base_config=TrainingConfig(
            train_file="data/train.jsonl",
            validation_file="data/validation.jsonl",
            test_file="data/test.jsonl",
        ),
        search_space={"train_batch_size": SweepParameter(values=[4], type="int")},
    )
    fold_specs = [
        FoldSpec(1, ("fold_2.jsonl",), "fold_1.jsonl", ("doc-a",)),
        FoldSpec(2, ("fold_1.jsonl",), "fold_2.jsonl", ("doc-b",)),
    ]
    monkeypatch.setattr("token_classification.sweep.prepare_cross_validation_folds", lambda *_args: fold_specs)
    calls = []

    def fake_run_pipeline(config, **kwargs):
        calls.append((config, kwargs))
        fold_index = int(config.run_name.rsplit("_", 1)[1])
        f1 = 0.6 if fold_index == 1 else 0.8
        run_dir = Path(config.output_dir) / config.run_name
        checkpoint = run_dir / "checkpoint-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_text("temporary", encoding="utf-8")
        (run_dir / "model.safetensors").write_text("must not remain", encoding="utf-8")
        write_json(
            run_dir / "run_summary.json",
            {
                "validation_metrics": {"eval_nervaluate_partial_micro_f1": f1},
                "completed_epoch": 3.0,
                "best_metric": f1,
            },
        )
        counts = {
            "correct": fold_index,
            "incorrect": 0,
            "partial": 0,
            "missed": 1,
            "spurious": 0,
            "actual": fold_index,
            "possible": fold_index + 1,
            "precision": 1.0,
            "recall": fold_index / (fold_index + 1),
            "f1": f1,
        }
        write_json(
            run_dir / "nervaluate" / "nervaluate_validation.json",
            {
                "labels": ["PERSON"],
                "rollups": {
                    "partial": {
                        "overall": {
                            "micro": counts,
                            "macro": {"precision": 1.0, "recall": counts["recall"], "f1": f1},
                        }
                    }
                },
                "per_tag_results": {
                    "PERSON": {
                        "gold_support": fold_index + 1,
                        "evaluable": True,
                        "scenarios": {"partial": counts},
                    }
                },
            },
        )
        return run_dir

    monkeypatch.setattr("token_classification.sweep.run_pipeline", fake_run_pipeline)

    sweep_root = run_sweep(sweep_config)

    trial_summary = json.loads((sweep_root / "trial_001" / "trial_summary.json").read_text(encoding="utf-8"))
    assert trial_summary["objective_mean"] == 0.7
    assert trial_summary["objective_std"] > 0
    assert len(calls) == 2
    assert all(call[1]["save_model"] is False for call in calls)
    assert all(call[1]["write_validation_nervaluate"] is True for call in calls)
    assert not list((sweep_root / "trial_001").glob("fold_*/checkpoint-*"))
    assert not list((sweep_root / "trial_001").glob("fold_*/model.safetensors"))
