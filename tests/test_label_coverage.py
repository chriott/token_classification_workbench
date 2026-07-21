from __future__ import annotations

import csv

from token_classification.config import TrainingConfig
from token_classification.label_coverage import build_label_coverage_report


def test_build_label_coverage_report_lists_missing_labels(tmp_path):
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"

    rows_by_path = {
        train_path: [
            {"text": "hello world", "spans": "[{'start': 0, 'end': 5, 'label': 'GREETING'}]"},
            {"text": "bye now", "spans": "[{'start': 0, 'end': 3, 'label': 'FAREWELL'}]"},
        ],
        validation_path: [
            {"text": "hello there", "spans": "[{'start': 0, 'end': 5, 'label': 'GREETING'}]"},
        ],
        test_path: [
            {"text": "bye there", "spans": "[{'start': 0, 'end': 3, 'label': 'FAREWELL'}]"},
            {"text": "wave now", "spans": "[{'start': 0, 'end': 4, 'label': 'GESTURE'}]"},
        ],
    }

    for path, rows in rows_by_path.items():
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["text", "spans"])
            writer.writeheader()
            writer.writerows(rows)

    config = TrainingConfig(
        train_file=str(train_path),
        validation_file=str(validation_path),
        test_file=str(test_path),
    )

    report = build_label_coverage_report(config)

    assert report["all_labels"] == ["FAREWELL", "GESTURE", "GREETING"]
    assert report["splits"]["train"]["missing_labels"] == ["GESTURE"]
    assert report["splits"]["validation"]["missing_labels"] == ["FAREWELL", "GESTURE"]
    assert report["splits"]["test"]["missing_labels"] == ["GREETING"]
