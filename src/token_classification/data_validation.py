from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import TrainingConfig
from .data import ensure_text_column, find_spans_column, load_csv_dataset, parse_spans_value
from .utils import write_json


def clean_span(span: Any):
    if not isinstance(span, dict):
        return None, "span_not_dict"

    label = span.get("label")
    start_raw = span.get("start")
    end_raw = span.get("end")
    if label in (None, ""):
        return None, "missing_label"

    try:
        start = int(start_raw)
        end = int(end_raw)
    except (TypeError, ValueError):
        return None, "non_integer_offsets"

    if start < 0 or end < 0 or end <= start:
        return None, "invalid_range"

    return {"start": start, "end": end, "label": str(label)}, None


def validate_dataset_file(path: str | Path, config: TrainingConfig, split_name: str, tokenizer=None) -> dict[str, Any]:
    raw_dataset = load_csv_dataset(str(path))
    dataset = ensure_text_column(raw_dataset, config.text_column)
    spans_column = find_spans_column(dataset.column_names, config.spans_column)
    if spans_column is None:
        raise ValueError(
            "Input data does not contain a spans column. "
            f"Expected '{config.spans_column}' or a supported fallback such as 'annotations'."
        )

    example_count = len(dataset)
    summary = {
        "split": split_name,
        "path": str(path),
        "row_count": example_count,
        "column_names": list(dataset.column_names),
        "text_column": "text",
        "spans_column": spans_column,
        "rows_with_empty_text": 0,
        "rows_with_parse_errors": 0,
        "rows_with_invalid_spans": 0,
        "rows_with_overlaps": 0,
        "rows_with_out_of_bounds_spans": 0,
        "rows_with_alignment_failures": 0,
        "total_spans": 0,
        "valid_spans": 0,
        "alignment_failures": 0,
        "max_text_length": 0,
        "avg_text_length": 0.0,
        "issue_counts": {
            "parse_errors": 0,
            "span_not_dict": 0,
            "missing_label": 0,
            "non_integer_offsets": 0,
            "invalid_range": 0,
            "out_of_bounds": 0,
            "alignment_failures": 0,
            "overlaps": 0,
        },
        "label_counts": {},
        "example_issues": [],
    }

    total_text_length = 0
    for index, example in enumerate(dataset):
        text = example.get("text", "") or ""
        text_length = len(text)
        total_text_length += text_length
        summary["max_text_length"] = max(summary["max_text_length"], text_length)
        if not text.strip():
            summary["rows_with_empty_text"] += 1

        raw_spans = example.get(spans_column)
        parsed_spans, parse_error = parse_spans_value(raw_spans)
        row_issue_codes = set()
        if parse_error is not None:
            summary["rows_with_parse_errors"] += 1
            summary["issue_counts"]["parse_errors"] += 1
            row_issue_codes.add("parse_errors")

        valid_spans = []
        for span in parsed_spans:
            summary["total_spans"] += 1
            cleaned_span, error_code = clean_span(span)
            if error_code is not None:
                summary["issue_counts"][error_code] += 1
                row_issue_codes.add(error_code)
                continue
            if cleaned_span["end"] > text_length or cleaned_span["start"] > text_length:
                summary["issue_counts"]["out_of_bounds"] += 1
                row_issue_codes.add("out_of_bounds")
                continue
            valid_spans.append(cleaned_span)
            summary["valid_spans"] += 1
            summary["label_counts"][cleaned_span["label"]] = summary["label_counts"].get(cleaned_span["label"], 0) + 1

        valid_spans.sort(key=lambda span: (span["start"], span["end"], span["label"]))
        previous_end = None
        for span in valid_spans:
            if previous_end is not None and span["start"] < previous_end:
                summary["issue_counts"]["overlaps"] += 1
                row_issue_codes.add("overlaps")
            previous_end = max(previous_end or span["end"], span["end"])

        if tokenizer is not None and valid_spans:
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=config.max_length,
                return_offsets_mapping=True,
                return_attention_mask=True,
            )
            failed_alignments = 0
            for span in valid_spans:
                token_start = tokenized.char_to_token(span["start"])
                token_end = tokenized.char_to_token(span["end"] - 1)
                if token_start is None or token_end is None:
                    failed_alignments += 1
            if failed_alignments:
                summary["alignment_failures"] += failed_alignments
                summary["issue_counts"]["alignment_failures"] += failed_alignments
                row_issue_codes.add("alignment_failures")

        if row_issue_codes:
            summary["rows_with_invalid_spans"] += 1
            if "out_of_bounds" in row_issue_codes:
                summary["rows_with_out_of_bounds_spans"] += 1
            if "overlaps" in row_issue_codes:
                summary["rows_with_overlaps"] += 1
            if "alignment_failures" in row_issue_codes:
                summary["rows_with_alignment_failures"] += 1
            if len(summary["example_issues"]) < 25:
                example_id = example.get("source_row_id", index)
                summary["example_issues"].append(
                    {
                        "row_index": index,
                        "example_id": example_id,
                        "issue_codes": sorted(row_issue_codes),
                    }
                )

    if example_count:
        summary["avg_text_length"] = total_text_length / example_count

    return summary


def validate_dataset_splits(
    config: TrainingConfig,
    split_names: tuple[str, ...] = ("train", "validation", "test"),
    check_alignment: bool = True,
) -> dict[str, Any]:
    tokenizer = None
    if check_alignment:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    split_to_path = {
        "train": config.train_file,
        "validation": config.validation_file,
        "test": config.test_file,
    }

    summaries = {}
    for split_name in split_names:
        split_path = split_to_path.get(split_name)
        if not split_path:
            continue
        summaries[split_name] = validate_dataset_file(split_path, config, split_name=split_name, tokenizer=tokenizer)

    return {
        "model_name": config.model_name,
        "max_length": config.max_length,
        "checked_alignment": check_alignment,
        "splits": summaries,
    }


def save_validation_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    return write_json(output_path, summary)
