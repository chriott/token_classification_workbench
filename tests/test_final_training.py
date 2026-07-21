from __future__ import annotations

from ipi_training.cli import build_parser
from ipi_training.config import TrainingConfig
from ipi_training.training import merge_training_splits_for_final_mode


def test_training_config_allows_missing_validation_for_final_training():
    config = TrainingConfig(
        train_file="data/train.csv",
        validation_file=None,
        test_file="data/test.csv",
    )

    config.validate(require_validation=False)


def test_cli_parses_final_train_command():
    parser = build_parser()

    args = parser.parse_args(["final-train", "--config", "configs/example.yaml"])

    assert args.command == "final-train"
    assert args.config == "configs/example.yaml"


def test_final_training_mode_merges_train_and_validation_in_memory():
    dataset_splits = {
        "train": ["train_a", "train_b"],
        "validation": ["val_a"],
        "test": ["test_a"],
    }

    merged = merge_training_splits_for_final_mode(
        dataset_splits,
        final_training_mode=True,
        concatenate_fn=lambda datasets: datasets[0] + datasets[1],
        dataset_dict_factory=dict,
    )

    assert merged["train"] == ["train_a", "train_b", "val_a"]
    assert "validation" not in merged
    assert merged["test"] == ["test_a"]
