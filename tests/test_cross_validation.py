from __future__ import annotations

import json

from token_classification.config import CrossValidationConfig, TrainingConfig
from token_classification.cross_validation import prepare_cross_validation_folds


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_prepare_cross_validation_folds_keeps_document_chunks_together(tmp_path):
    rows = []
    for document_index in range(10):
        label = "PERSON" if document_index % 2 == 0 else "PLACE"
        for chunk_index in range(2):
            rows.append(
                {
                    "document_id": f"doc-{document_index}",
                    "chunk": chunk_index,
                    "text": label,
                    "spans": [{"start": 0, "end": len(label), "label": label}],
                }
            )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_jsonl(train_path, rows[:12])
    write_jsonl(validation_path, rows[12:])
    training_config = TrainingConfig(
        train_file=str(train_path),
        validation_file=str(validation_path),
        test_file="test.jsonl",
    )
    cv_config = CrossValidationConfig(folds=5, seed=137, group_column="document_id")

    folds = prepare_cross_validation_folds(training_config, cv_config, tmp_path / "folds")

    validation_groups = [group for fold in folds for group in fold.validation_groups]
    assert len(folds) == 5
    assert sorted(validation_groups) == [f"doc-{index}" for index in range(10)]
    assert len(set(validation_groups)) == 10
    for fold in folds:
        fold_rows = [json.loads(line) for line in open(fold.validation_file, encoding="utf-8")]
        groups_in_file = {row["document_id"] for row in fold_rows}
        assert groups_in_file == set(fold.validation_groups)
        assert all(sum(row["document_id"] == group for row in fold_rows) == 2 for group in groups_in_file)


def test_cross_validation_rejects_labels_present_in_only_one_document_group(tmp_path):
    rows = [
        {
            "document_id": f"doc-{index}",
            "text": "text",
            "spans": [{"start": 0, "end": 4, "label": "RARE" if index == 0 else "COMMON"}],
        }
        for index in range(5)
    ]
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_jsonl(train_path, rows[:3])
    write_jsonl(validation_path, rows[3:])
    training_config = TrainingConfig(
        train_file=str(train_path),
        validation_file=str(validation_path),
        test_file="test.jsonl",
    )

    try:
        prepare_cross_validation_folds(
            training_config,
            CrossValidationConfig(folds=5, seed=137, group_column="document_id"),
            tmp_path / "folds",
        )
    except ValueError as exc:
        assert "at least two document groups" in str(exc)
        assert "RARE" in str(exc)
    else:
        raise AssertionError("Expected unsupported rare labels to be rejected.")


def test_cross_validation_can_explicitly_exclude_smoke_test_labels(tmp_path):
    rows = [
        {
            "document_id": f"doc-{index}",
            "text": "text",
            "spans": [
                {"start": 0, "end": 4, "label": "COMMON"},
                *([{"start": 0, "end": 4, "label": "SINGLETON"}] if index == 0 else []),
            ],
        }
        for index in range(4)
    ]
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_jsonl(train_path, rows[:3])
    write_jsonl(validation_path, rows[3:])
    training_config = TrainingConfig(
        train_file=str(train_path),
        validation_file=str(validation_path),
        test_file="test.jsonl",
    )

    folds = prepare_cross_validation_folds(
        training_config,
        CrossValidationConfig(
            folds=2,
            seed=137,
            group_column="document_id",
            excluded_labels=("SINGLETON",),
        ),
        tmp_path / "folds",
    )

    fold_rows = [
        json.loads(line)
        for fold in folds
        for line in open(fold.validation_file, encoding="utf-8")
    ]
    assert all(span["label"] != "SINGLETON" for row in fold_rows for span in row["spans"])
    manifest = json.loads((tmp_path / "folds" / "fold_manifest.json").read_text(encoding="utf-8"))
    assert manifest["excluded_labels"] == ["SINGLETON"]
    assert manifest["label_group_support"] == {"COMMON": 4}
