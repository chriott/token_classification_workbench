from token_classification.data import _make_parse_spans_fn, parse_spans_value


def test_parse_spans_string():
    parse_spans = _make_parse_spans_fn("spans")
    example = {"spans": "[{'start': 1, 'end': 4, 'label': 'PERSON'}]"}

    parsed = parse_spans(example)

    assert parsed["spans"] == [{"start": 1, "end": 4, "label": "PERSON"}]


def test_parse_spans_with_smart_quotes_falls_back_cleanly():
    parse_spans = _make_parse_spans_fn("spans")
    example = {"spans": "[{“start”: 1, “end”: 4, “label”: “PERSON”}]"}

    parsed = parse_spans(example)

    assert parsed["spans"] == [{"start": 1, "end": 4, "label": "PERSON"}]


def test_parse_spans_value_rejects_non_list_payloads():
    parsed, error = parse_spans_value("{'start': 1, 'end': 4, 'label': 'PERSON'}")

    assert parsed == []
    assert error is not None


def test_parse_spans_value_normalizes_annotation_style_offsets():
    parsed, error = parse_spans_value(
        [
            {"begin": 3, "end": 9, "label": "TIME", "text_span": "foobar"},
            {"begin": 12, "text_span": "abc", "label": "AGE"},
        ]
    )

    assert error is None
    assert parsed == [
        {"start": 3, "end": 9, "label": "TIME"},
        {"start": 12, "end": 15, "label": "AGE"},
    ]
