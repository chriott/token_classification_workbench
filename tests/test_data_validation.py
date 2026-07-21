from __future__ import annotations

import csv

from token_classification.config import TrainingConfig
from token_classification.data_validation import validate_dataset_file


def test_validate_dataset_file_reports_basic_span_issues(tmp_path):
    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "spans"])
        writer.writeheader()
        writer.writerow({"text": "hello world", "spans": "[{'start': 0, 'end': 5, 'label': 'GREETING'}]"})
        writer.writerow({"text": "short", "spans": "[{'start': 0, 'end': 20, 'label': 'TOO_LONG'}]"})
        writer.writerow({"text": "bad", "spans": "not a valid span payload"})

    config = TrainingConfig(
        train_file=str(csv_path),
        validation_file=str(csv_path),
        test_file=str(csv_path),
    )

    summary = validate_dataset_file(csv_path, config, split_name="train", tokenizer=None)

    assert summary["row_count"] == 3
    assert summary["valid_spans"] == 1
    assert summary["rows_with_parse_errors"] == 1
    assert summary["rows_with_out_of_bounds_spans"] == 1
    assert summary["issue_counts"]["out_of_bounds"] == 1
    assert summary["label_counts"] == {"GREETING": 1}
