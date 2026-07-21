from ipi_training.evaluation import classify_span_sets
from ipi_training.labels import bio_to_entities, bio_to_spans


def test_bio_to_spans_groups_adjacent_i_tags():
    labels = ["B-IPI-NAME", "I-IPI-NAME", "O", "B-PHI-DATE"]
    offsets = [(0, 4), (5, 8), (0, 0), (10, 20)]

    spans = bio_to_spans(labels, offsets)

    assert spans == [
        {"start": 0, "end": 8, "label": "IPI-NAME"},
        {"start": 10, "end": 20, "label": "PHI-DATE"},
    ]


def test_bio_to_entities_and_classification():
    token_spans = [(0, 4), (5, 8), (10, 20)]
    gold_labels = ["B-IPI-NAME", "I-IPI-NAME", "B-PHI-DATE"]
    predicted_labels = ["B-IPI-NAME", "I-IPI-NAME", "O"]

    gold_entities = bio_to_entities(token_spans, gold_labels)
    predicted_entities = bio_to_entities(token_spans, predicted_labels)
    true_positive_spans, false_positive_spans, false_negative_spans = classify_span_sets(
        predicted_entities, gold_entities
    )

    assert gold_entities == [
        {"label": "IPI-NAME", "start": 0, "end": 8},
        {"label": "PHI-DATE", "start": 10, "end": 20},
    ]
    assert predicted_entities == [{"label": "IPI-NAME", "start": 0, "end": 8}]
    assert true_positive_spans == [{"label": "IPI-NAME", "start": 0, "end": 8}]
    assert false_positive_spans == []
    assert false_negative_spans == [{"label": "PHI-DATE", "start": 10, "end": 20}]
