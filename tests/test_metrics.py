import csv
import json
from types import SimpleNamespace

import numpy as np

from token_classification.evaluation import classify_span_sets, compute_micro_macro, save_test_predictions
from token_classification.labels import LabelSchema, bio_to_entities, bio_to_spans


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def with_format(self, _format):
        return self

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]


def test_compute_micro_macro_includes_macro_precision_and_recall():
    rows = [
        {
            "correct": 8,
            "missed": 2,
            "spurious": 2,
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8,
        },
        {
            "correct": 1,
            "missed": 3,
            "spurious": 0,
            "precision": 1.0,
            "recall": 0.25,
            "f1": 0.4,
        },
    ]

    rollup = compute_micro_macro(rows)

    assert rollup == {
        "micro": {
            "precision": 9 / 11,
            "recall": 9 / 14,
            "f1": 2 * (9 / 11) * (9 / 14) / ((9 / 11) + (9 / 14)),
            "true_positives": 9,
            "false_positives": 2,
            "false_negatives": 5,
        },
        "macro": {
            "precision": 0.9,
            "recall": 0.525,
            "f1": (0.8 + 0.4) / 2,
            "label_count": 2,
        },
    }


def test_bio_to_spans_groups_adjacent_i_tags():
    labels = ["B-PERSON", "I-PERSON", "O", "B-DATE"]
    offsets = [(0, 4), (5, 8), (0, 0), (10, 20)]

    spans = bio_to_spans(labels, offsets)

    assert spans == [
        {"start": 0, "end": 8, "label": "PERSON"},
        {"start": 10, "end": 20, "label": "DATE"},
    ]


def test_bio_to_entities_and_classification():
    token_spans = [(0, 4), (5, 8), (10, 20)]
    gold_labels = ["B-PERSON", "I-PERSON", "B-DATE"]
    predicted_labels = ["B-PERSON", "I-PERSON", "O"]

    gold_entities = bio_to_entities(token_spans, gold_labels)
    predicted_entities = bio_to_entities(token_spans, predicted_labels)
    true_positive_spans, false_positive_spans, false_negative_spans = classify_span_sets(
        predicted_entities, gold_entities
    )

    assert gold_entities == [
        {"label": "PERSON", "start": 0, "end": 8},
        {"label": "DATE", "start": 10, "end": 20},
    ]
    assert predicted_entities == [{"label": "PERSON", "start": 0, "end": 8}]
    assert true_positive_spans == [{"label": "PERSON", "start": 0, "end": 8}]
    assert false_positive_spans == []
    assert false_negative_spans == [{"label": "DATE", "start": 10, "end": 20}]


def test_save_test_predictions_exports_only_configured_metadata(tmp_path):
    trainer = SimpleNamespace(
        predict=lambda _dataset: SimpleNamespace(
            predictions=np.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]]]),
            label_ids=np.array([[-100, 1, -100]]),
        )
    )
    tokenized = FakeDataset([{"offset_mapping": [(0, 0), (0, 3), (0, 0)]}])
    raw = FakeDataset(
        [
            {
                "text": "Ada",
                "spans": [{"start": 0, "end": 3, "label": "PERSON"}],
                "document_id": "doc-1",
                "unused": "not-exported",
            }
        ]
    )
    schema = LabelSchema(
        labels=["PERSON"],
        bio_labels=["O", "B-PERSON", "I-PERSON"],
        label_to_id={"O": 0, "B-PERSON": 1, "I-PERSON": 2},
        id_to_label={0: "O", 1: "B-PERSON", 2: "I-PERSON"},
    )

    jsonl_path, csv_path = save_test_predictions(
        trainer,
        tokenized,
        raw,
        tmp_path,
        schema,
        metadata_fields=("document_id",),
    )

    detailed_record = json.loads(jsonl_path.read_text(encoding="utf-8"))
    assert detailed_record["metadata"] == {"document_id": "doc-1"}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["document_id"] == "doc-1"
    assert "unused" not in row
