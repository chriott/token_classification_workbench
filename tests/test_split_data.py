from __future__ import annotations

import csv
import json

from token_classification.split_data import _chunk_record, split_input_data


class FakeWhitespaceTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True, truncation=False):
        offsets = []
        cursor = 0
        for token in text.split():
            start = text.index(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        return {"offset_mapping": offsets}


def test_split_input_data_accepts_jsonl_annotations_and_writes_csv(tmp_path):
    input_path = tmp_path / "input.jsonl"
    records = [
        {
            "doc_key": f"doc_{index}",
            "text": f"Example {index} in Berlin",
            "annotations": [{"begin": 13, "end": 19, "label": "GEOLOCATION"}],
        }
        for index in range(10)
    ]
    with input_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    summary = split_input_data(
        input_file=input_path,
        output_dir=tmp_path / "splits",
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        seed=7,
        output_format="csv",
    )

    assert summary["counts"] == {"train": 8, "validation": 1, "test": 1}

    train_path = tmp_path / "splits" / "train.csv"
    assert train_path.exists()

    with train_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8
    assert "spans" in rows[0]
    assert "annotations" not in rows[0]
    assert json.loads(rows[0]["spans"]) == [{"start": 13, "end": 19, "label": "GEOLOCATION"}]


def test_split_input_data_supports_label_signature_stratification(tmp_path):
    input_path = tmp_path / "input.jsonl"
    records = []
    for label in ("AGE", "TIME", "LIFESTYLE"):
        for index in range(10):
            records.append(
                {
                    "doc_key": f"{label}_{index}",
                    "text": f"{label} text {index}",
                    "annotations": [{"begin": 0, "end": len(label), "label": label}],
                }
            )
    with input_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    summary = split_input_data(
        input_file=input_path,
        output_dir=tmp_path / "splits",
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        seed=11,
        output_format="jsonl",
        stratify_by="label_signature",
    )

    assert summary["document_counts"] == {"train": 24, "validation": 3, "test": 3}

    train_rows = [json.loads(line) for line in (tmp_path / "splits" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    validation_rows = [
        json.loads(line) for line in (tmp_path / "splits" / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    test_rows = [json.loads(line) for line in (tmp_path / "splits" / "test.jsonl").read_text(encoding="utf-8").splitlines()]

    for rows, expected_count in ((train_rows, 8), (validation_rows, 1), (test_rows, 1)):
        label_counts = {}
        for row in rows:
            label = row["spans"][0]["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        assert label_counts == {"AGE": expected_count, "TIME": expected_count, "LIFESTYLE": expected_count}


def test_split_input_data_supports_primary_label_stratification(tmp_path):
    input_path = tmp_path / "input.jsonl"
    records = []
    for label in ("AGE", "TIME", "LIFESTYLE"):
        for index in range(10):
            records.append(
                {
                    "doc_key": f"{label}_{index}",
                    "text": f"{label} text {index}",
                    "annotations": [
                        {"begin": 0, "end": len(label), "label": label},
                        {"begin": len(label) + 1, "end": len(label) + 5, "label": label},
                    ],
                }
            )
    with input_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    summary = split_input_data(
        input_file=input_path,
        output_dir=tmp_path / "splits",
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        seed=11,
        output_format="jsonl",
        stratify_by="primary_label",
    )

    assert summary["document_counts"] == {"train": 24, "validation": 3, "test": 3}


def test_chunk_record_uses_overlap_to_preserve_boundary_crossing_entity():
    tokenizer = FakeWhitespaceTokenizer()
    text = "aa bb cc dd ee ff"
    span_start = text.index("dd")
    span_end = text.index("ee") + len("ee")
    record = {
        "text": text,
        "spans": [{"start": span_start, "end": span_end, "label": "ENTITY"}],
        "_source_row_index": 0,
    }

    chunks = _chunk_record(
        record,
        tokenizer=tokenizer,
        chunk_max_length=4,
        chunk_stride=1,
    )

    assert len(chunks) >= 2
    assert chunks[0]["spans"] == []
    assert any(chunk["spans"] == [{"start": 0, "end": 5, "label": "ENTITY"}] for chunk in chunks)


def test_split_input_data_can_chunk_after_splitting(tmp_path):
    input_path = tmp_path / "input.jsonl"
    records = [
        {
            "doc_key": f"doc_{index}",
            "text": "one two three four five six",
            "annotations": [{"begin": 4, "end": 7, "label": "TIME"}],
        }
        for index in range(10)
    ]
    with input_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    summary = split_input_data(
        input_file=input_path,
        output_dir=tmp_path / "splits",
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        seed=3,
        output_format="csv",
        chunk_max_length=4,
        chunk_stride=2,
        chunk_tokenizer=FakeWhitespaceTokenizer(),
    )

    assert summary["chunking"]["enabled"] is True
    assert summary["document_counts"] == {"train": 8, "validation": 1, "test": 1}
    assert summary["counts"]["train"] > summary["document_counts"]["train"]

    with (tmp_path / "splits" / "train.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "chunk_index" in rows[0]
    assert "token_len" in rows[0]
    assert "char_start" in rows[0]
    assert "char_end" in rows[0]


def test_split_input_data_iterative_multilabel_saves_manifest(tmp_path):
    input_path = tmp_path / "input.jsonl"
    records = []
    for index in range(12):
        labels = ["AGE", "TIME"]
        if index % 2 == 0:
            labels.append("LIFESTYLE")
        records.append(
            {
                "doc_key": f"doc_{index}",
                "text": f"doc {index}",
                "annotations": [
                    {"begin": offset * 2, "end": offset * 2 + len(label), "label": label}
                    for offset, label in enumerate(labels)
                ],
            }
        )
    with input_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    summary = split_input_data(
        input_file=input_path,
        output_dir=tmp_path / "splits",
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        seed=25,
        output_format="jsonl",
        stratify_by="iterative_multilabel",
    )

    manifest_path = tmp_path / "splits" / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary["split_manifest"] == str(manifest_path)
    assert manifest["stratify_by"] == "iterative_multilabel"
    assert set(manifest["splits"]) == {"train", "validation", "test"}


def test_split_input_data_can_reuse_manifest_for_chunking(tmp_path):
    input_path = tmp_path / "input.jsonl"
    records = [
        {
            "doc_key": f"doc_{index}",
            "text": "one two three four five six",
            "annotations": [{"begin": 4, "end": 7, "label": "TIME"}],
        }
        for index in range(10)
    ]
    with input_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    manifest_summary = split_input_data(
        input_file=input_path,
        output_dir=tmp_path / "manifest_only",
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        seed=25,
        output_format="jsonl",
        stratify_by="iterative_multilabel",
    )

    summary = split_input_data(
        input_file=input_path,
        output_dir=tmp_path / "chunked",
        output_format="csv",
        split_manifest_in=manifest_summary["split_manifest"],
        chunk_max_length=4,
        chunk_stride=2,
        chunk_tokenizer=FakeWhitespaceTokenizer(),
    )

    assert summary["document_counts"] == {"train": 8, "validation": 1, "test": 1}
    with (tmp_path / "chunked" / "train.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["doc_key"] for row in rows}.issubset({f"doc_{index}" for index in range(10)})


def test_split_input_data_constrained_min_labels_writes_stats(tmp_path):
    input_path = tmp_path / "input.jsonl"
    records = []
    for index in range(9):
        labels = ["TIME"]
        if index < 6:
            labels.append("AGE")
        if index < 3:
            labels.append("LIFESTYLE")
        records.append(
            {
                "doc_key": f"doc_{index}",
                "text": f"doc {index}",
                "annotations": [
                    {"begin": offset * 2, "end": offset * 2 + len(label), "label": label}
                    for offset, label in enumerate(labels)
                ],
            }
        )
    with input_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    summary = split_input_data(
        input_file=input_path,
        output_dir=tmp_path / "constrained",
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        seed=25,
        output_format="jsonl",
        stratify_by="constrained_min_labels",
        min_label_presence=1,
    )

    stats_path = tmp_path / "constrained" / "label_split_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert summary["label_split_stats"] == str(stats_path)
    assert stats["all_labels"] == ["AGE", "LIFESTYLE", "TIME"]
    assert stats["coverage_summary"]["min_label_presence"] == 1
    assert "TIME" in stats["splits"]["validation"]["present_labels"]
