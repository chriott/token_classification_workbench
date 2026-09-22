from __future__ import annotations

from token_classification.data import filter_excluded_labels


class FakeDataset(list):
    def map(self, function, **_kwargs):
        return FakeDataset([{**example, **function(example)} for example in self])


def test_filter_excluded_labels_removes_only_matching_spans_and_keeps_rows():
    dataset = FakeDataset(
        [
            {
                "text": "example one",
                "spans": [
                    {"start": 0, "end": 7, "label": "KEEP"},
                    {"start": 8, "end": 11, "label": "RARE"},
                ],
            },
            {
                "text": "example two",
                "spans": [{"start": 0, "end": 7, "label": "RARE"}],
            },
        ]
    )

    filtered = filter_excluded_labels(dataset, ("RARE",))

    assert len(filtered) == 2
    assert [span["label"] for span in filtered[0]["spans"]] == ["KEEP"]
    assert filtered[1]["spans"] == []
