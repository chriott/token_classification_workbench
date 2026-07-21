from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .data import find_spans_column, find_text_column, parse_spans_value
from .utils import ensure_directory


def _validate_split_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("train_ratio, validation_ratio, and test_ratio must all be greater than zero.")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}.")


def _validate_chunking_args(chunk_max_length: int | None, chunk_stride: int) -> None:
    if chunk_max_length is None:
        return
    if chunk_max_length <= 0:
        raise ValueError("chunk_max_length must be greater than zero.")
    if chunk_stride < 0:
        raise ValueError("chunk_stride must be zero or greater.")
    if chunk_stride >= chunk_max_length:
        raise ValueError("chunk_stride must be smaller than chunk_max_length.")


def _normalize_record(example: dict[str, Any], spans_column: str, row_index: int) -> dict[str, Any]:
    parsed_spans, parse_error = parse_spans_value(example.get(spans_column))
    if parse_error is not None:
        identifier = example.get("doc_key") or example.get("source_row_id") or "unknown"
        raise ValueError(f"Could not parse spans for record {identifier}: {parse_error}")

    normalized = dict(example)
    normalized["text"] = example.get("text", "") or ""
    normalized["spans"] = parsed_spans
    normalized["_source_row_index"] = row_index
    normalized["_record_id"] = str(example.get("doc_key") or example.get("source_row_id") or row_index)
    if spans_column != "spans":
        normalized.pop(spans_column, None)
    return normalized


def _read_input_records(input_file: str | Path) -> list[dict[str, Any]]:
    path = Path(input_file)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list):
            raise ValueError(f"Expected top-level list in JSON input file {path}")
        return loaded
    raise ValueError(f"Unsupported input file format: {suffix}")


def _load_records(input_file: str | Path, text_column: str | None, spans_column: str | None) -> list[dict[str, Any]]:
    raw_records = _read_input_records(input_file)
    if not raw_records:
        raise ValueError(f"Input file {input_file} does not contain any records.")
    if not all(isinstance(record, dict) for record in raw_records):
        raise ValueError("All input records must be objects/dictionaries.")

    first_record = raw_records[0]
    columns = list(first_record.keys())
    source_text_column = text_column or ("text" if "text" in columns else find_text_column(columns))
    if source_text_column is None:
        raise ValueError(
            "Input data does not contain a text column and no fallback text source was found.\n"
            "Add a 'text' column or set --text-column."
        )

    resolved_spans_column = find_spans_column(columns, spans_column)
    if resolved_spans_column is None:
        raise ValueError(
            "Input data does not contain a spans column. "
            f"Expected '{spans_column or 'spans'}' or a supported fallback such as 'annotations'."
        )

    normalized_records = []
    for index, example in enumerate(raw_records):
        record = dict(example)
        if source_text_column != "text":
            record["text"] = example.get(source_text_column, "") or ""
        normalized_records.append(_normalize_record(record, resolved_spans_column, row_index=index))
    return normalized_records


def _largest_remainder_counts(size: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw_counts = [size * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remainder = size - sum(counts)
    remainders = sorted(
        ((raw_counts[index] - counts[index], index) for index in range(len(raw_counts))),
        reverse=True,
    )
    for _, index in remainders[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _stratification_key(record: dict[str, Any], stratify_by: str, stratify_column: str | None) -> str | None:
    if stratify_column:
        value = record.get(stratify_column)
        return "__MISSING__" if value in (None, "") else str(value)
    if stratify_by == "primary_label":
        label_counts: dict[str, int] = {}
        for span in record.get("spans", []):
            label = span.get("label")
            if label:
                label_counts[str(label)] = label_counts.get(str(label), 0) + 1
        if not label_counts:
            return "__NO_LABELS__"
        return sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    if stratify_by == "label_signature":
        labels = sorted({str(span.get("label")) for span in record.get("spans", []) if span.get("label")})
        return "|".join(labels) if labels else "__NO_LABELS__"
    return None


def _labels_for_record(record: dict[str, Any]) -> list[str]:
    return sorted({str(span.get("label")) for span in record.get("spans", []) if span.get("label")})


def _iterative_multilabel_split(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    ratios = (train_ratio, validation_ratio, test_ratio)
    split_names = ("train", "validation", "test")
    rng = random.Random(seed)

    target_docs = dict(zip(split_names, _largest_remainder_counts(len(records), ratios)))
    split_records = {name: [] for name in split_names}
    remaining_doc_capacity = dict(target_docs)

    record_labels = {record["_record_id"]: _labels_for_record(record) for record in records}
    label_totals = Counter(label for labels in record_labels.values() for label in labels)
    target_label_counts = {
        label: dict(zip(split_names, _largest_remainder_counts(total, ratios)))
        for label, total in label_totals.items()
    }
    remaining_label_needs = {
        label: dict(targets) for label, targets in target_label_counts.items()
    }

    pending = {record["_record_id"]: record for record in records}

    def choose_split(labels: list[str]) -> str:
        best_split = None
        best_score = None
        for split_name in split_names:
            if remaining_doc_capacity[split_name] <= 0:
                continue
            primary_need = max((remaining_label_needs.get(label, {}).get(split_name, 0) for label in labels), default=0)
            total_need = sum(remaining_label_needs.get(label, {}).get(split_name, 0) for label in labels)
            score = (primary_need, total_need, remaining_doc_capacity[split_name], -len(split_records[split_name]))
            if best_score is None or score > best_score:
                best_score = score
                best_split = split_name
        if best_split is None:
            best_split = max(split_names, key=lambda split_name: (remaining_doc_capacity[split_name], -len(split_records[split_name])))
        return best_split

    while pending:
        remaining_label_counts = Counter(
            label
            for record_id in pending
            for label in record_labels[record_id]
        )
        if remaining_label_counts:
            rarest_count = min(remaining_label_counts.values())
            rarest_labels = [label for label, count in remaining_label_counts.items() if count == rarest_count]
            selected_label = rng.choice(sorted(rarest_labels))
            candidate_records = [
                pending[record_id]
                for record_id in pending
                if selected_label in record_labels[record_id]
            ]
        else:
            candidate_records = list(pending.values())

        rng.shuffle(candidate_records)
        candidate_records.sort(
            key=lambda record: (
                len(record_labels[record["_record_id"]]),
                sum(label_totals.get(label, 0) for label in record_labels[record["_record_id"]]),
            ),
            reverse=True,
        )
        record = candidate_records[0]
        record_id = record["_record_id"]
        labels = record_labels[record_id]
        split_name = choose_split(labels)
        split_records[split_name].append(record)
        remaining_doc_capacity[split_name] -= 1
        for label in labels:
            if label in remaining_label_needs:
                remaining_label_needs[label][split_name] = max(0, remaining_label_needs[label][split_name] - 1)
        pending.pop(record_id)

    for rows in split_records.values():
        rng.shuffle(rows)

    if not all(split_records[name] for name in split_names):
        raise ValueError(
            "Iterative multilabel split produced an empty split. "
            "Use a larger dataset or different ratios."
        )
    return split_records


def _build_split_manifest(
    records: list[dict[str, Any]],
    split_records: dict[str, list[dict[str, Any]]],
    *,
    input_file: str | Path,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_by: str,
    stratify_column: str | None,
) -> dict[str, Any]:
    all_labels = sorted({label for record in records for label in _labels_for_record(record)})
    return {
        "input_file": str(input_file),
        "seed": seed,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "stratify_by": stratify_by,
        "stratify_column": stratify_column,
        "record_id_field": "_record_id",
        "all_labels": all_labels,
        "splits": {
            split_name: [record["_record_id"] for record in split_rows]
            for split_name, split_rows in split_records.items()
        },
    }


def _apply_split_manifest(records: list[dict[str, Any]], manifest_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    split_ids = manifest.get("splits")
    if not isinstance(split_ids, dict):
        raise ValueError("Split manifest does not contain a valid 'splits' mapping.")

    records_by_id = {record["_record_id"]: record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("Split manifest application requires unique record identifiers.")

    assigned_ids = set()
    split_records: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "validation", "test"):
        ids = split_ids.get(split_name)
        if not isinstance(ids, list):
            raise ValueError(f"Split manifest is missing list for split '{split_name}'.")
        missing_ids = [record_id for record_id in ids if record_id not in records_by_id]
        if missing_ids:
            raise ValueError(f"Split manifest references unknown record ids for {split_name}: {missing_ids[:5]}")
        split_records[split_name] = [records_by_id[record_id] for record_id in ids]
        overlap = assigned_ids.intersection(ids)
        if overlap:
            raise ValueError(f"Split manifest assigns the same record to multiple splits: {sorted(overlap)[:5]}")
        assigned_ids.update(ids)

    unassigned = sorted(set(records_by_id) - assigned_ids)
    if unassigned:
        raise ValueError(f"Split manifest does not assign all input records. Unassigned examples: {unassigned[:5]}")
    return split_records


def _largest_remainder_counts_with_minimum(
    size: int,
    ratios: tuple[float, float, float],
    minimum: int,
) -> tuple[int, int, int]:
    if size == 0:
        return (0, 0, 0)
    counts = list(_largest_remainder_counts(size, ratios))
    if size >= minimum * 3:
        counts = [max(count, minimum) for count in counts]
        overflow = sum(counts) - size
        while overflow > 0:
            candidates = [index for index, count in enumerate(counts) if count > minimum]
            if not candidates:
                break
            index = max(candidates, key=lambda candidate: counts[candidate])
            counts[index] -= 1
            overflow -= 1
    return counts[0], counts[1], counts[2]


def _constrained_min_label_split(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    min_label_presence: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    ratios = (train_ratio, validation_ratio, test_ratio)
    split_names = ("train", "validation", "test")
    rng = random.Random(seed)

    target_docs = dict(zip(split_names, _largest_remainder_counts(len(records), ratios)))
    remaining_doc_capacity = dict(target_docs)
    split_records = {name: [] for name in split_names}

    record_labels = {record["_record_id"]: _labels_for_record(record) for record in records}
    label_to_record_ids: dict[str, list[str]] = {}
    for record_id, labels in record_labels.items():
        for label in labels:
            label_to_record_ids.setdefault(label, []).append(record_id)

    pending = {record["_record_id"]: record for record in records}
    doc_label_counts = {split_name: Counter() for split_name in split_names}

    label_totals = {label: len(record_ids) for label, record_ids in label_to_record_ids.items()}
    target_label_counts = {
        label: dict(zip(split_names, _largest_remainder_counts_with_minimum(total, ratios, min_label_presence)))
        for label, total in label_totals.items()
    }
    feasible_labels = {
        label for label, total in label_totals.items() if total >= min_label_presence * len(split_names)
    }

    def assign_record(record_id: str, split_name: str) -> None:
        record = pending.pop(record_id)
        split_records[split_name].append(record)
        remaining_doc_capacity[split_name] -= 1
        for label in record_labels[record_id]:
            doc_label_counts[split_name][label] += 1

    sorted_labels = sorted(label_to_record_ids, key=lambda label: (label_totals[label], label))
    for label in sorted_labels:
        if label not in feasible_labels:
            continue
        for split_name in split_names:
            while doc_label_counts[split_name][label] < min_label_presence:
                candidates = [
                    record_id for record_id in label_to_record_ids[label]
                    if record_id in pending and remaining_doc_capacity[split_name] > 0
                ]
                if not candidates:
                    break
                candidates.sort(
                    key=lambda record_id: (
                        sum(
                            1
                            for candidate_label in record_labels[record_id]
                            if candidate_label in feasible_labels
                            and doc_label_counts[split_name][candidate_label] < min_label_presence
                        ),
                        -sum(label_totals.get(candidate_label, 0) for candidate_label in record_labels[record_id]),
                        len(record_labels[record_id]),
                        record_id,
                    ),
                    reverse=True,
                )
                assign_record(candidates[0], split_name)

    remaining_label_needs = {
        label: {
            split_name: max(0, target_label_counts[label][split_name] - doc_label_counts[split_name][label])
            for split_name in split_names
        }
        for label in label_to_record_ids
    }

    def choose_split_for_remaining(record_id: str) -> str:
        labels = record_labels[record_id]
        best_split = None
        best_score = None
        for split_name in split_names:
            if remaining_doc_capacity[split_name] <= 0:
                continue
            needed_labels = sum(remaining_label_needs[label][split_name] for label in labels)
            unmet_minimum_labels = sum(
                1
                for label in labels
                if label in feasible_labels and doc_label_counts[split_name][label] < min_label_presence
            )
            score = (unmet_minimum_labels, needed_labels, remaining_doc_capacity[split_name], -len(split_records[split_name]))
            if best_score is None or score > best_score:
                best_score = score
                best_split = split_name
        if best_split is None:
            best_split = max(split_names, key=lambda split_name: (remaining_doc_capacity[split_name], -len(split_records[split_name])))
        return best_split

    while pending:
        remaining_label_counts = Counter(
            label
            for record_id in pending
            for label in record_labels[record_id]
        )
        if remaining_label_counts:
            rarest_count = min(remaining_label_counts.values())
            rarest_labels = [label for label, count in remaining_label_counts.items() if count == rarest_count]
            selected_label = rng.choice(sorted(rarest_labels))
            candidate_ids = [record_id for record_id in pending if selected_label in record_labels[record_id]]
        else:
            candidate_ids = list(pending)
        rng.shuffle(candidate_ids)
        candidate_ids.sort(
            key=lambda record_id: (
                sum(max(remaining_label_needs[label].values()) for label in record_labels[record_id]),
                len(record_labels[record_id]),
                -sum(label_totals.get(label, 0) for label in record_labels[record_id]),
                record_id,
            ),
            reverse=True,
        )
        record_id = candidate_ids[0]
        split_name = choose_split_for_remaining(record_id)
        assign_record(record_id, split_name)
        for label in record_labels[record_id]:
            remaining_label_needs[label][split_name] = max(0, remaining_label_needs[label][split_name] - 1)

    for rows in split_records.values():
        rng.shuffle(rows)

    coverage_summary = {
        "min_label_presence": min_label_presence,
        "feasible_labels": sorted(feasible_labels),
        "infeasible_labels": sorted(label for label in label_to_record_ids if label not in feasible_labels),
        "target_label_doc_counts": target_label_counts,
        "actual_label_doc_counts": {
            split_name: dict(sorted(doc_label_counts[split_name].items()))
            for split_name in split_names
        },
    }
    return split_records, coverage_summary


def _build_label_split_stats(
    original_records: list[dict[str, Any]],
    split_records: dict[str, list[dict[str, Any]]],
    *,
    coverage_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_label_span_counts = Counter()
    original_label_doc_counts = Counter()
    for record in original_records:
        labels_in_doc = set()
        for span in record.get("spans", []):
            label = span.get("label")
            if label:
                original_label_span_counts[str(label)] += 1
                labels_in_doc.add(str(label))
        for label in labels_in_doc:
            original_label_doc_counts[label] += 1

    all_labels = sorted(original_label_span_counts)
    splits: dict[str, dict[str, Any]] = {}
    for split_name, rows in split_records.items():
        label_span_counts = Counter()
        label_doc_counts = Counter()
        for record in rows:
            labels_in_doc = set()
            for span in record.get("spans", []):
                label = span.get("label")
                if label:
                    label_span_counts[str(label)] += 1
                    labels_in_doc.add(str(label))
            for label in labels_in_doc:
                label_doc_counts[label] += 1
        splits[split_name] = {
            "document_count": len(rows),
            "present_labels": sorted(label_span_counts),
            "missing_labels": [label for label in all_labels if label not in label_span_counts],
            "label_span_counts": dict(sorted(label_span_counts.items())),
            "label_doc_counts": dict(sorted(label_doc_counts.items())),
        }

    stats = {
        "all_labels": all_labels,
        "all_label_count": len(all_labels),
        "original_label_span_counts": dict(sorted(original_label_span_counts.items())),
        "original_label_doc_counts": dict(sorted(original_label_doc_counts.items())),
        "splits": splits,
    }
    if coverage_summary is not None:
        stats["coverage_summary"] = coverage_summary
    return stats


def _split_records(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_by: str = "none",
    stratify_column: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    _validate_split_ratios(train_ratio, validation_ratio, test_ratio)
    if len(records) < 3:
        raise ValueError("At least 3 records are required to create train/validation/test splits.")

    ratios = (train_ratio, validation_ratio, test_ratio)
    rng = random.Random(seed)
    if stratify_by == "none" and not stratify_column:
        shuffled = list(records)
        rng.shuffle(shuffled)
        train_end, validation_count, _ = _largest_remainder_counts(len(shuffled), ratios)
        validation_end = train_end + validation_count
        if train_end == 0 or validation_end <= train_end or validation_end >= len(shuffled):
            raise ValueError(
                "Split ratios produced an empty split. "
                "Adjust the ratios or use a larger dataset."
            )
        return {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:validation_end],
            "test": shuffled[validation_end:],
        }
    if stratify_by == "iterative_multilabel":
        return _iterative_multilabel_split(
            records,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = _stratification_key(record, stratify_by=stratify_by, stratify_column=stratify_column)
        grouped.setdefault(key or "__DEFAULT__", []).append(record)

    split_records = {"train": [], "validation": [], "test": []}
    for group_records in grouped.values():
        shuffled = list(group_records)
        rng.shuffle(shuffled)
        train_count, validation_count, test_count = _largest_remainder_counts(len(shuffled), ratios)
        cursor = 0
        split_records["train"].extend(shuffled[cursor : cursor + train_count])
        cursor += train_count
        split_records["validation"].extend(shuffled[cursor : cursor + validation_count])
        cursor += validation_count
        split_records["test"].extend(shuffled[cursor : cursor + test_count])

    for rows in split_records.values():
        rng.shuffle(rows)

    if not all(split_records[name] for name in ("train", "validation", "test")):
        raise ValueError(
            "Stratified split produced an empty split. "
            "Use a larger dataset, different ratios, or disable stratification."
        )
    return split_records


def _build_chunk_tokenizer(tokenizer_model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)


def _find_token_index_for_char(offsets: list[tuple[int, int]], char_position: int) -> int | None:
    for index, (start, end) in enumerate(offsets):
        if start <= char_position < end:
            return index
    return None


def _resolve_chunk_content_length(tokenizer, chunk_max_length: int, chunk_stride: int) -> tuple[int, int]:
    count_special_tokens = getattr(tokenizer, "num_special_tokens_to_add", None)
    if callable(count_special_tokens):
        try:
            special_token_count = int(count_special_tokens(pair=False))
        except TypeError:
            special_token_count = int(count_special_tokens())
    else:
        special_token_count = 0

    content_max_length = chunk_max_length - special_token_count
    if content_max_length < 1:
        raise ValueError(
            "chunk_max_length must leave room for at least one content token after "
            f"the tokenizer adds {special_token_count} special token(s)."
        )
    if chunk_stride >= content_max_length:
        raise ValueError(
            "chunk_stride must be smaller than the effective content-token capacity "
            f"({content_max_length}) after reserving special tokens."
        )
    return content_max_length, special_token_count


def _chunk_record(record: dict[str, Any], *, tokenizer, chunk_max_length: int, chunk_stride: int) -> list[dict[str, Any]]:
    content_max_length, _ = _resolve_chunk_content_length(tokenizer, chunk_max_length, chunk_stride)
    text = record.get("text", "") or ""
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    raw_offsets = tokenized.get("offset_mapping", [])
    offsets = [(int(start), int(end)) for start, end in raw_offsets if int(end) > int(start)]
    if not offsets:
        chunk = dict(record)
        chunk["chunk_index"] = 0
        chunk["char_start"] = 0
        chunk["char_end"] = len(text)
        chunk["token_len"] = 0
        chunk["annotation_count"] = 0
        return [chunk]

    chunks: list[dict[str, Any]] = []
    start_token = 0
    chunk_index = 0
    while start_token < len(offsets):
        end_token = min(start_token + content_max_length, len(offsets))
        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]

        chunk_spans = []
        overlapping_end_spans = []
        for span in record.get("spans", []):
            span_start = int(span.get("start", 0))
            span_end = int(span.get("end", 0))
            if span_start >= char_start and span_end <= char_end:
                chunk_spans.append(
                    {
                        "start": span_start - char_start,
                        "end": span_end - char_start,
                        "label": span.get("label"),
                    }
                )
            elif span_start < char_end < span_end:
                overlapping_end_spans.append(span)

        chunk = {key: value for key, value in record.items() if key != "spans"}
        chunk["text"] = text[char_start:char_end]
        chunk["spans"] = chunk_spans
        chunk["chunk_index"] = chunk_index
        chunk["char_start"] = char_start
        chunk["char_end"] = char_end
        chunk["token_len"] = end_token - start_token
        chunk["annotation_count"] = len(chunk_spans)
        chunk["source_row_id"] = record.get("source_row_id", record.get("_source_row_index"))
        chunks.append(chunk)

        if end_token >= len(offsets):
            break

        next_start = end_token - chunk_stride if chunk_stride else end_token
        if overlapping_end_spans:
            earliest_char = min(int(span.get("start", 0)) for span in overlapping_end_spans)
            token_index = _find_token_index_for_char(offsets, earliest_char)
            if token_index is not None:
                next_start = min(next_start, token_index)
        if next_start <= start_token:
            next_start = start_token + 1
        start_token = next_start
        chunk_index += 1

    return chunks


def _chunk_records(
    records: list[dict[str, Any]],
    *,
    chunk_max_length: int,
    chunk_stride: int,
    tokenizer,
) -> list[dict[str, Any]]:
    chunked_records = []
    for record in records:
        chunked_records.extend(
            _chunk_record(
                record,
                tokenizer=tokenizer,
                chunk_max_length=chunk_max_length,
                chunk_stride=chunk_stride,
            )
        )
    return chunked_records


def _prepare_output_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _write_csv(path: Path, records: list[dict[str, Any]]) -> Path:
    if not records:
        raise ValueError(f"Cannot write empty split to {path}")
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = _prepare_output_record(dict(record))
            row["spans"] = json.dumps(row.get("spans", []), ensure_ascii=False)
            writer.writerow(row)
    return path


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_prepare_output_record(record), ensure_ascii=False) + "\n")
    return path


def write_split_files(
    output_dir: str | Path,
    split_records: dict[str, list[dict[str, Any]]],
    *,
    output_format: str,
) -> dict[str, Path]:
    target_dir = ensure_directory(output_dir)
    written_paths: dict[str, Path] = {}
    for split_name, records in split_records.items():
        suffix = ".csv" if output_format == "csv" else ".jsonl"
        path = target_dir / f"{split_name}{suffix}"
        if output_format == "csv":
            written_paths[split_name] = _write_csv(path, records)
        elif output_format == "jsonl":
            written_paths[split_name] = _write_jsonl(path, records)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    return written_paths


def split_input_data(
    *,
    input_file: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    text_column: str | None = None,
    spans_column: str | None = None,
    output_format: str = "csv",
    stratify_by: str = "none",
    stratify_column: str | None = None,
    split_manifest_in: str | Path | None = None,
    min_label_presence: int = 1,
    chunk_max_length: int | None = None,
    chunk_stride: int = 0,
    chunk_tokenizer_model: str | None = None,
    chunk_tokenizer=None,
) -> dict[str, Any]:
    _validate_chunking_args(chunk_max_length, chunk_stride)
    records = _load_records(input_file, text_column=text_column, spans_column=spans_column)
    manifest_path = Path(output_dir) / "split_manifest.json"
    coverage_summary: dict[str, Any] | None = None
    if split_manifest_in is not None:
        split_records = _apply_split_manifest(records, split_manifest_in)
        manifest = json.loads(Path(split_manifest_in).read_text(encoding="utf-8"))
    else:
        if stratify_by == "constrained_min_labels":
            split_records, coverage_summary = _constrained_min_label_split(
                records,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                seed=seed,
                min_label_presence=min_label_presence,
            )
        else:
            split_records = _split_records(
                records,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                seed=seed,
                stratify_by=stratify_by,
                stratify_column=stratify_column,
            )
        manifest = _build_split_manifest(
            records,
            split_records,
            input_file=input_file,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratify_by=stratify_by,
            stratify_column=stratify_column,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    document_counts = {split_name: len(rows) for split_name, rows in split_records.items()}
    chunk_content_length: int | None = None
    chunk_special_token_count: int | None = None
    if chunk_max_length is not None:
        tokenizer = chunk_tokenizer
        if tokenizer is None:
            if not chunk_tokenizer_model:
                raise ValueError("chunk_tokenizer_model must be set when chunk_max_length is enabled.")
            tokenizer = _build_chunk_tokenizer(chunk_tokenizer_model)
        chunk_content_length, chunk_special_token_count = _resolve_chunk_content_length(
            tokenizer,
            chunk_max_length,
            chunk_stride,
        )
        split_records = {
            split_name: _chunk_records(
                rows,
                chunk_max_length=chunk_max_length,
                chunk_stride=chunk_stride,
                tokenizer=tokenizer,
            )
            for split_name, rows in split_records.items()
        }

    output_paths = write_split_files(output_dir, split_records, output_format=output_format)
    label_stats = _build_label_split_stats(records, split_records, coverage_summary=coverage_summary)
    label_stats_path = Path(output_dir) / "label_split_stats.json"
    label_stats_path.write_text(json.dumps(label_stats, indent=2), encoding="utf-8")
    return {
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "output_format": output_format,
        "seed": seed,
        "stratify_by": stratify_by,
        "stratify_column": stratify_column,
        "min_label_presence": min_label_presence,
        "split_manifest": str(manifest_path if split_manifest_in is None else split_manifest_in),
        "label_split_stats": str(label_stats_path),
        "document_counts": document_counts,
        "counts": {split_name: len(rows) for split_name, rows in split_records.items()},
        "chunking": {
            "enabled": chunk_max_length is not None,
            "chunk_max_length": chunk_max_length,
            "content_max_length": chunk_content_length,
            "special_token_count": chunk_special_token_count,
            "chunk_stride": chunk_stride,
            "chunk_tokenizer_model": chunk_tokenizer_model,
        },
        "output_files": {split_name: str(path) for split_name, path in output_paths.items()},
    }
