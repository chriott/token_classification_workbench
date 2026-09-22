from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import CrossValidationConfig, TrainingConfig
from .data import parse_spans_value
from .split_data import _load_records, _write_jsonl
from .utils import ensure_directory, write_json


@dataclass(frozen=True)
class FoldSpec:
    index: int
    train_files: tuple[str, ...]
    validation_file: str
    validation_groups: tuple[str, ...]


def _summary(values_by_fold: dict[int, float]) -> dict[str, object]:
    ordered = {str(index): values_by_fold[index] for index in sorted(values_by_fold)}
    values = list(ordered.values())
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
        "values_by_fold": ordered,
    }


def _metrics_from_counts(scenario: str, counts: dict[str, int]) -> dict[str, int | float]:
    actual = counts["correct"] + counts["incorrect"] + counts["partial"] + counts["spurious"]
    possible = counts["correct"] + counts["incorrect"] + counts["partial"] + counts["missed"]
    partial_credit = 0.5 * counts["partial"] if scenario in {"ent_type", "partial"} else 0.0
    true_positives = counts["correct"] + partial_credit
    precision = true_positives / actual if actual else 0.0
    recall = true_positives / possible if possible else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **counts,
        "actual": actual,
        "possible": possible,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_fold_nervaluate_results(results_by_fold: dict[int, dict[str, object]]) -> dict[str, object]:
    if not results_by_fold:
        raise ValueError("At least one fold result is required.")
    fold_indices = sorted(results_by_fold)
    scenarios = sorted(results_by_fold[fold_indices[0]].get("rollups", {}))
    labels = sorted(results_by_fold[fold_indices[0]].get("labels", []))
    count_fields = ("correct", "incorrect", "partial", "missed", "spurious")
    output: dict[str, object] = {"fold_count": len(fold_indices), "scenarios": {}}

    for scenario in scenarios:
        overall = {}
        for averaging in ("micro", "macro"):
            overall[averaging] = {
                metric: _summary(
                    {
                        fold_index: float(
                            results_by_fold[fold_index]["rollups"][scenario]["overall"][averaging][metric]
                        )
                        for fold_index in fold_indices
                    }
                )
                for metric in ("precision", "recall", "f1")
            }

        pooled_micro_counts = {field: 0 for field in count_fields}
        for fold_index in fold_indices:
            micro = results_by_fold[fold_index]["rollups"][scenario]["overall"]["micro"]
            for field in count_fields:
                pooled_micro_counts[field] += int(micro[field])
        pooled_micro = _metrics_from_counts(scenario, pooled_micro_counts)

        label_output = {}
        pooled_label_metrics = []
        for label in labels:
            metric_values = {metric: {} for metric in ("precision", "recall", "f1")}
            pooled_counts = {field: 0 for field in count_fields}
            gold_support = 0
            for fold_index in fold_indices:
                label_result = results_by_fold[fold_index]["per_tag_results"].get(label)
                if not label_result:
                    continue
                gold_support += int(label_result.get("gold_support", 0))
                scenario_result = label_result.get("scenarios", {}).get(scenario)
                if scenario_result is None:
                    continue
                for field in count_fields:
                    pooled_counts[field] += int(scenario_result[field])
                if label_result.get("evaluable", False):
                    for metric in metric_values:
                        metric_values[metric][fold_index] = float(scenario_result[metric])
            pooled = _metrics_from_counts(scenario, pooled_counts) if gold_support > 0 else None
            if pooled is not None:
                pooled_label_metrics.append(pooled)
            label_output[label] = {
                "gold_support": gold_support,
                "evaluated_folds": len(metric_values["f1"]),
                "fold_metrics": {
                    metric: _summary(values) for metric, values in metric_values.items() if values
                },
                "pooled": pooled,
            }

        pooled_macro = None
        if pooled_label_metrics:
            pooled_macro = {
                metric: statistics.mean(float(row[metric]) for row in pooled_label_metrics)
                for metric in ("precision", "recall", "f1")
            }
            pooled_macro["label_count"] = len(pooled_label_metrics)

        output["scenarios"][scenario] = {
            "fold_metrics": overall,
            "pooled": {"micro": pooled_micro, "macro": pooled_macro},
            "labels": label_output,
        }
    return output


def _source_paths(path: str | Sequence[str] | None) -> list[str]:
    if path is None:
        return []
    if isinstance(path, str):
        return [part.strip() for part in path.split(",") if part.strip()]
    return [str(part) for part in path]


def _load_development_records(config: TrainingConfig) -> list[dict[str, Any]]:
    paths = _source_paths(config.train_file) + _source_paths(config.validation_file)
    records = []
    for path in paths:
        records.extend(_load_records(path, text_column=config.text_column, spans_column=config.spans_column))
    if not records:
        raise ValueError("Cross-validation requires non-empty training and validation data.")
    return records


def _group_labels(records: Sequence[dict[str, Any]]) -> set[str]:
    labels = set()
    for record in records:
        spans = record.get("spans", [])
        if not isinstance(spans, list):
            spans, error = parse_spans_value(spans)
            if error:
                raise ValueError(error)
        labels.update(str(span["label"]) for span in spans if span.get("label"))
    return labels


def assign_groups_to_folds(
    grouped_records: dict[str, list[dict[str, Any]]],
    *,
    folds: int,
    seed: int,
    stratify_by: str,
) -> list[list[str]]:
    if len(grouped_records) < folds:
        raise ValueError(f"Cannot create {folds} folds from only {len(grouped_records)} document groups.")

    rng = random.Random(seed)
    group_ids = sorted(grouped_records)
    fold_groups = [[] for _ in range(folds)]
    target_sizes = [len(group_ids) // folds + (1 if index < len(group_ids) % folds else 0) for index in range(folds)]

    if stratify_by == "none":
        rng.shuffle(group_ids)
        cursor = 0
        for fold_index, target_size in enumerate(target_sizes):
            fold_groups[fold_index].extend(group_ids[cursor : cursor + target_size])
            cursor += target_size
        return fold_groups

    labels_by_group = {group_id: _group_labels(grouped_records[group_id]) for group_id in group_ids}
    label_totals = Counter(label for labels in labels_by_group.values() for label in labels)
    fold_label_counts = [Counter() for _ in range(folds)]
    shuffled_order = list(group_ids)
    rng.shuffle(shuffled_order)
    shuffled_order.sort(
        key=lambda group_id: (
            min((label_totals[label] for label in labels_by_group[group_id]), default=len(group_ids) + 1),
            -len(labels_by_group[group_id]),
        )
    )

    for group_id in shuffled_order:
        labels = labels_by_group[group_id]
        candidates = [index for index in range(folds) if len(fold_groups[index]) < target_sizes[index]]
        best_score = None
        best_folds = []
        for fold_index in candidates:
            label_need = sum(label_totals[label] / folds - fold_label_counts[fold_index][label] for label in labels)
            score = (label_need, target_sizes[fold_index] - len(fold_groups[fold_index]))
            if best_score is None or score > best_score:
                best_score = score
                best_folds = [fold_index]
            elif score == best_score:
                best_folds.append(fold_index)
        chosen_fold = rng.choice(best_folds)
        fold_groups[chosen_fold].append(group_id)
        fold_label_counts[chosen_fold].update(labels)

    for groups in fold_groups:
        groups.sort()
    return fold_groups


def prepare_cross_validation_folds(
    training_config: TrainingConfig,
    cross_validation: CrossValidationConfig,
    output_dir: str | Path,
) -> list[FoldSpec]:
    cross_validation.validate()
    records = _load_development_records(training_config)
    excluded_labels = set(cross_validation.excluded_labels)
    if excluded_labels:
        available_labels = _group_labels(records)
        unknown_labels = sorted(excluded_labels - available_labels)
        if unknown_labels:
            raise ValueError(f"Cross-validation excluded_labels are not present in the data: {unknown_labels}.")
        records = [
            {
                **record,
                "spans": [
                    span
                    for span in record.get("spans", [])
                    if str(span.get("label")) not in excluded_labels
                ],
            }
            for record in records
        ]
    grouped_records: dict[str, list[dict[str, Any]]] = {}
    for row_index, record in enumerate(records):
        group_value = record.get(cross_validation.group_column)
        if group_value in (None, ""):
            raise ValueError(
                f"Record {row_index} has no value for cross-validation group column "
                f"'{cross_validation.group_column}'."
            )
        grouped_records.setdefault(str(group_value), []).append(record)

    all_labels = sorted({label for rows in grouped_records.values() for label in _group_labels(rows)})
    label_group_support = {
        label: sum(label in _group_labels(rows) for rows in grouped_records.values())
        for label in all_labels
    }
    unsupported_labels = sorted(label for label, support in label_group_support.items() if support < 2)
    if unsupported_labels:
        raise ValueError(
            "Cross-validation requires every label in at least two document groups; "
            f"insufficient labels: {unsupported_labels}."
        )

    fold_groups = assign_groups_to_folds(
        grouped_records,
        folds=cross_validation.folds,
        seed=cross_validation.seed,
        stratify_by=cross_validation.stratify_by,
    )
    target_dir = ensure_directory(output_dir)
    shard_paths = []
    for fold_index, validation_groups in enumerate(fold_groups, start=1):
        shard_records = [record for group_id in validation_groups for record in grouped_records[group_id]]
        shard_path = _write_jsonl(target_dir / f"fold_{fold_index}.jsonl", shard_records)
        shard_paths.append(shard_path)

    manifest = {
        "folds": cross_validation.folds,
        "seed": cross_validation.seed,
        "group_column": cross_validation.group_column,
        "stratify_by": cross_validation.stratify_by,
        "excluded_labels": sorted(excluded_labels),
        "group_count": len(grouped_records),
        "record_count": len(records),
        "label_group_support": label_group_support,
        "labels_with_fewer_groups_than_folds": sorted(
            label for label, support in label_group_support.items() if support < cross_validation.folds
        ),
        "fold_assignments": {
            f"fold_{index}": groups for index, groups in enumerate(fold_groups, start=1)
        },
        "fold_files": [str(path) for path in shard_paths],
    }
    write_json(target_dir / "fold_manifest.json", manifest)

    return [
        FoldSpec(
            index=index,
            train_files=tuple(str(path) for path in shard_paths if path != validation_path),
            validation_file=str(validation_path),
            validation_groups=tuple(fold_groups[index - 1]),
        )
        for index, validation_path in enumerate(shard_paths, start=1)
    ]
