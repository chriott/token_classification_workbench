from __future__ import annotations

from ipi_training.config import TrainingConfig, dump_config_yaml
from ipi_training.prediction import infer_prediction_settings


def test_infer_prediction_settings_uses_saved_training_config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dump_config_yaml(
        model_dir / "config_used.yaml",
        TrainingConfig(
            train_file="data/train.csv",
            validation_file="data/validation.csv",
            test_file="data/test.csv",
            max_length=384,
            text_column="note_text",
        ),
    )

    max_length, text_column = infer_prediction_settings(model_dir, max_length=None, text_column=None)

    assert max_length == 384
    assert text_column == "note_text"
