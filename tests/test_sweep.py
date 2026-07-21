import json
import random

from ipi_training.config import SweepConfig, SweepParameter, TrainingConfig
from ipi_training.sweep import enumerate_grid_trial_overrides, run_sweep, sample_parameter_value, sample_trial_overrides
from ipi_training.utils import write_json


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
        seed=42,
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
        seed=42,
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
        seed=42,
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
        seed=42,
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

    state = {"calls": 0}

    def fake_run_pipeline(config, run_test_evaluation=False):
        state["calls"] += 1
        run_dir = tmp_path / "outputs" / "demo_incremental" / config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if state["calls"] == 2:
            raise RuntimeError("boom")
        write_json(
            run_dir / "run_summary.json",
            {"validation_metrics": {"eval_f1_macro": 0.7}},
        )
        return run_dir

    monkeypatch.setattr("ipi_training.sweep.run_pipeline", fake_run_pipeline)

    sweep_root = run_sweep(sweep_config)
    leaderboard_path = sweep_root / "leaderboard.csv"
    summary_path = sweep_root / "summary.json"

    assert leaderboard_path.exists()
    assert summary_path.exists()

    leaderboard_rows = leaderboard_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(leaderboard_rows) == 3
    assert "trial_001" in leaderboard_rows[1]
    assert "trial_002" in leaderboard_rows[2]
    assert "failed" in leaderboard_rows[2]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["completed_trials"] == 1
    assert summary["failed_trials"] == 1
    assert summary["best_trial"]["run_name"] == "trial_001"
