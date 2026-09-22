from __future__ import annotations

import ast
import contextlib
import io
import os
from pathlib import Path
from typing import Sequence

from .config import TrainingConfig
from .utils import ensure_directory


def find_text_column(columns: Sequence[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in ("text", "note_text", "document", "doc", "notes", "content"):
        if candidate in lowered:
            return lowered[candidate]
    for candidate in ("TEXT", "Text"):
        if candidate in columns:
            return candidate
    return None


def find_spans_column(columns: Sequence[str], preferred: str | None = None) -> str | None:
    if preferred and preferred in columns:
        return preferred
    for candidate in ("spans", "annotations"):
        if candidate in columns:
            return candidate
    return None


def ensure_text_column(dataset, text_column: str | None):
    if text_column and text_column in dataset.column_names:
        source_column = text_column
    else:
        source_column = "text" if "text" in dataset.column_names else find_text_column(dataset.column_names)
        if source_column is None:
            raise ValueError(
                "CSV does not contain a text column and no fallback text source was found.\n"
                "Add a 'text' column or set text_column in your YAML config."
            )
    if source_column != "text":
        dataset = dataset.rename_column(source_column, "text")
    return dataset


def normalize_span_record(span):
    if not isinstance(span, dict):
        return span
    normalized = dict(span)
    if "start" not in normalized and "begin" in normalized:
        normalized["start"] = normalized.get("begin")
    if "end" not in normalized:
        if "stop" in normalized:
            normalized["end"] = normalized.get("stop")
        elif "text_span" in normalized and normalized.get("begin") is not None:
            normalized["end"] = int(normalized["begin"]) + len(str(normalized["text_span"]))
    if "end" not in normalized and "begin" in normalized and "text_span" in normalized:
        normalized["end"] = int(normalized["begin"]) + len(str(normalized["text_span"]))
    return {
        "start": normalized.get("start"),
        "end": normalized.get("end"),
        "label": normalized.get("label"),
    }


def normalize_optional_string_columns(dataset, optional_columns: Sequence[str]):
    for column in optional_columns:
        if column in dataset.column_names:
            dataset = dataset.map(
                lambda example, col=column: {col: "" if example[col] is None else str(example[col])},
                desc=f"normalizing column {column}",
            )
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    from datasets import Value

                dataset = dataset.cast_column(column, Value("string"))
            except (ImportError, TypeError, ValueError):
                pass
    return dataset


def parse_spans_value(spans_value):
    if isinstance(spans_value, str):
        try:
            parsed_value = ast.literal_eval(spans_value.strip())
        except Exception:
            cleaned = spans_value.replace("“", '"').replace("”", '"').replace("’", "'")
            try:
                parsed_value = ast.literal_eval(cleaned)
            except Exception:
                return [], "Could not parse spans string."
    elif isinstance(spans_value, list):
        parsed_value = spans_value
    elif spans_value in (None, ""):
        return [], None
    else:
        return [], f"Unsupported spans value type: {type(spans_value).__name__}"

    if parsed_value is None:
        return [], None
    if not isinstance(parsed_value, list):
        return [], f"Parsed spans value is {type(parsed_value).__name__}, expected list."
    return [normalize_span_record(span) for span in parsed_value], None


def _make_parse_spans_fn(spans_column: str):
    def parse(example):
        parsed_spans, _ = parse_spans_value(example.get(spans_column))
        example["spans"] = parsed_spans
        return example

    return parse


def load_csv_dataset(path: str | Sequence[str]):
    if isinstance(path, str):
        path_parts = [part.strip() for part in path.split(",") if part.strip()]
    else:
        path_parts = list(path)

    data_files = {"data": path_parts if len(path_parts) > 1 else path_parts[0]}
    suffixes = {Path(part).suffix.lower() for part in path_parts}
    if not suffixes:
        raise ValueError("At least one data file path must be provided.")
    if len(suffixes) != 1:
        raise ValueError(f"Mixed input formats are not supported in one dataset load: {sorted(suffixes)}")
    suffix = next(iter(suffixes))
    if suffix == ".csv":
        dataset_name = "csv"
    elif suffix in (".json", ".jsonl"):
        dataset_name = "json"
    else:
        raise ValueError(f"Unsupported input file format: {suffix}")
    cache_dir = ensure_directory(
        Path(os.environ.get("TOKEN_CLASSIFICATION_CACHE_DIR", ".cache/huggingface/datasets"))
    )
    with contextlib.redirect_stderr(io.StringIO()):
        from datasets import load_dataset
    return load_dataset(dataset_name, data_files=data_files, cache_dir=str(cache_dir))["data"]


def filter_excluded_labels(dataset, excluded_labels: Sequence[str]):
    excluded = frozenset(str(label) for label in excluded_labels)
    if not excluded:
        return dataset
    return dataset.map(
        lambda example: {
            "spans": [
                span
                for span in example.get("spans", [])
                if str(span.get("label")) not in excluded
            ]
        },
        desc="excluding configured labels",
    )


def _label_counts(dataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in dataset:
        for span in example.get("spans", []):
            label = span.get("label")
            if label in (None, ""):
                continue
            label_name = str(label)
            counts[label_name] = counts.get(label_name, 0) + 1
    return counts


def load_and_prepare_dataset(
    path: str | Sequence[str],
    config: TrainingConfig,
    *,
    apply_label_exclusions: bool = True,
):
    loaded = load_csv_dataset(path)
    loaded = ensure_text_column(loaded, config.text_column)
    spans_column = find_spans_column(loaded.column_names, config.spans_column)
    if spans_column is None:
        raise ValueError(
            "Input data does not contain a spans column. "
            f"Expected '{config.spans_column}' or a supported fallback such as 'annotations'."
        )
    loaded = loaded.map(_make_parse_spans_fn(spans_column))
    loaded = normalize_optional_string_columns(loaded, config.optional_string_columns)
    if apply_label_exclusions:
        loaded = filter_excluded_labels(loaded, config.excluded_labels)
    return loaded


def load_prediction_dataset(path: str | Sequence[str], text_column: str | None, optional_columns: Sequence[str] = ()):
    loaded = load_csv_dataset(path)
    loaded = ensure_text_column(loaded, text_column)
    loaded = normalize_optional_string_columns(loaded, optional_columns)
    return loaded


def load_dataset_splits(config: TrainingConfig):
    with contextlib.redirect_stderr(io.StringIO()):
        from datasets import DatasetDict
    dataset = DatasetDict()
    dataset["train"] = load_and_prepare_dataset(config.train_file, config, apply_label_exclusions=False)
    if config.validation_file:
        dataset["validation"] = load_and_prepare_dataset(
            config.validation_file,
            config,
            apply_label_exclusions=False,
        )
    dataset["test"] = load_and_prepare_dataset(config.test_file, config, apply_label_exclusions=False)

    if config.excluded_labels:
        raw_counts = {split_name: _label_counts(split) for split_name, split in dataset.items()}
        available_labels = {
            label
            for split_counts in raw_counts.values()
            for label in split_counts
        }
        unknown_labels = sorted(set(config.excluded_labels) - available_labels)
        if unknown_labels:
            raise ValueError(f"Training excluded_labels are not present in the data: {unknown_labels}.")

        print("\nConfigured label exclusions:")
        for label in config.excluded_labels:
            counts = ", ".join(
                f"{split_name}={split_counts.get(label, 0)}"
                for split_name, split_counts in raw_counts.items()
            )
            print(f"- {label}: {counts}")
        for split_name in dataset:
            dataset[split_name] = filter_excluded_labels(dataset[split_name], config.excluded_labels)
    return dataset
