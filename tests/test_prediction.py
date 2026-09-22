from __future__ import annotations

from token_classification.config import TrainingConfig, dump_config_yaml
from token_classification.prediction import aggregate_chunk_predictions, infer_prediction_settings


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
            text_column="content",
        ),
    )

    max_length, text_column = infer_prediction_settings(model_dir, max_length=None, text_column=None)

    assert max_length == 384
    assert text_column == "content"


def test_aggregate_chunk_predictions_restores_offsets_and_deduplicates_overlap():
    records = [
        {
            "index": 0,
            "metadata": {"document_id": "doc.txt", "chunk_index": 0, "char_start": 0, "char_end": 11},
            "text": "hello world",
            "predicted_spans": [{"start": 6, "end": 11, "label": "PLACE", "text": "world"}],
        },
        {
            "index": 1,
            "metadata": {"document_id": "doc.txt", "chunk_index": 1, "char_start": 6, "char_end": 17},
            "text": "world again",
            "predicted_spans": [
                {"start": 0, "end": 5, "label": "PLACE", "text": "world"},
                {"start": 6, "end": 11, "label": "EVENT", "text": "again"},
            ],
        },
    ]

    documents = aggregate_chunk_predictions(records)

    assert len(documents) == 1
    assert documents[0]["document_id"] == "doc.txt"
    assert documents[0]["text"] == "hello world again"
    assert documents[0]["predicted_spans"] == [
        {"start": 6, "end": 11, "label": "PLACE", "text": "world"},
        {"start": 12, "end": 17, "label": "EVENT", "text": "again"},
    ]
