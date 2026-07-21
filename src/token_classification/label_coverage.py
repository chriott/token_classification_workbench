from __future__ import annotations

from pathlib import Path

from .config import TrainingConfig
from .data import load_and_prepare_dataset
from .utils import write_json


def build_label_coverage_report(
    config: TrainingConfig,
    split_names: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, object]:
    split_to_path = {
        "train": config.train_file,
        "validation": config.validation_file,
        "test": config.test_file,
    }

    split_counts: dict[str, dict[str, int]] = {}
    split_label_sets: dict[str, set[str]] = {}

    for split_name in split_names:
        split_path = split_to_path.get(split_name)
        if not split_path:
            continue

        dataset = load_and_prepare_dataset(split_path, config)
        label_counts: dict[str, int] = {}
        for example in dataset:
            for span in example.get("spans", []):
                label = span.get("label")
                if not label:
                    continue
                label_name = str(label)
                label_counts[label_name] = label_counts.get(label_name, 0) + 1

        split_counts[split_name] = dict(sorted(label_counts.items()))
        split_label_sets[split_name] = set(label_counts)

    all_labels = sorted(set().union(*split_label_sets.values())) if split_label_sets else []

    splits: dict[str, dict[str, object]] = {}
    for split_name, label_counts in split_counts.items():
        present_labels = sorted(split_label_sets[split_name])
        missing_labels = [label for label in all_labels if label not in split_label_sets[split_name]]
        splits[split_name] = {
            "label_count": len(present_labels),
            "span_count": sum(label_counts.values()),
            "present_labels": present_labels,
            "missing_labels": missing_labels,
            "label_counts": label_counts,
        }

    return {
        "all_labels": all_labels,
        "all_label_count": len(all_labels),
        "splits": splits,
    }


def save_label_coverage_report(report: dict[str, object], output_path: str | Path) -> Path:
    return write_json(output_path, report)
