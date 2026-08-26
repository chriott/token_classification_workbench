from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .labels import LabelSchema, bio_to_entities, bio_to_spans
from .utils import write_json


NERVALUATE_SCENARIOS = ("strict", "ent_type", "partial")


def evaluation_result_to_dict(result) -> Dict[str, int | float]:
    return {
        "correct": int(result.correct),
        "incorrect": int(result.incorrect),
        "partial": int(result.partial),
        "missed": int(result.missed),
        "spurious": int(result.spurious),
        "actual": int(result.actual),
        "possible": int(result.possible),
        "precision": float(result.precision),
        "recall": float(result.recall),
        "f1": float(result.f1),
    }


def compute_micro_macro(
    scenario: str,
    overall_result: Mapping[str, int | float],
    per_label_results: Sequence[Mapping[str, int | float]],
):
    if not per_label_results:
        return None

    partial_credit = 0.5 * float(overall_result["partial"]) if scenario in {"ent_type", "partial"} else 0.0
    true_positives = float(overall_result["correct"]) + partial_credit
    false_positives = float(overall_result["actual"]) - true_positives
    false_negatives = float(overall_result["possible"]) - true_positives
    label_count = len(per_label_results)
    return {
        "micro": {
            **overall_result,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
        "macro": {
            "precision": sum(float(row["precision"]) for row in per_label_results) / label_count,
            "recall": sum(float(row["recall"]) for row in per_label_results) / label_count,
            "f1": sum(float(row["f1"]) for row in per_label_results) / label_count,
            "label_count": label_count,
        },
    }


def compute_nervaluate_metrics(true_labels: Sequence[Sequence[str]], predicted_labels: Sequence[Sequence[str]], tags):
    from nervaluate import Evaluator

    evaluator = Evaluator(true_labels, predicted_labels, tags=list(tags), loader="list")
    evaluated = evaluator.evaluate()
    metrics = {}
    for scenario in NERVALUATE_SCENARIOS:
        overall_result = evaluated["overall"].get(scenario)
        if overall_result is None:
            continue
        per_label_results = []
        for label in tags:
            label_result = evaluated["entities"].get(label, {}).get(scenario)
            if label_result is not None and label_result.possible > 0:
                per_label_results.append(evaluation_result_to_dict(label_result))
        rollup = compute_micro_macro(
            scenario,
            evaluation_result_to_dict(overall_result),
            per_label_results,
        )
        if rollup is None:
            continue
        for averaging in ("micro", "macro"):
            for metric in ("precision", "recall", "f1"):
                metrics[f"nervaluate_{scenario}_{averaging}_{metric}"] = rollup[averaging][metric]
    return metrics


def build_compute_metrics_fn(schema: LabelSchema):
    import numpy as np
    from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

    def compute_metrics(prediction_output):
        predictions, labels = prediction_output
        predictions = np.argmax(predictions, axis=2)

        true_labels = [[schema.id_to_label[label] for label in row if label != -100] for row in labels]
        true_predictions = [
            [schema.id_to_label[prediction] for prediction, label in zip(pred_row, label_row) if label != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]

        precision_micro = precision_score(true_labels, true_predictions, average="micro")
        recall_micro = recall_score(true_labels, true_predictions, average="micro")
        f1_micro = f1_score(true_labels, true_predictions, average="micro")
        precision_macro = precision_score(true_labels, true_predictions, average="macro")
        recall_macro = recall_score(true_labels, true_predictions, average="macro")
        f1_macro = f1_score(true_labels, true_predictions, average="macro")

        metrics = {
            "eval_accuracy": accuracy_score(true_labels, true_predictions),
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
        }
        metrics.update(compute_nervaluate_metrics(true_labels, true_predictions, schema.labels))
        return metrics

    return compute_metrics


def persist_label_schema(schema: LabelSchema, output_dir: str | Path) -> None:
    target_dir = Path(output_dir)
    write_json(target_dir / "label2id.json", schema.label_to_id)
    write_json(target_dir / "id2label.json", {str(key): value for key, value in schema.id_to_label.items()})
    write_json(target_dir / "bio_labels.json", schema.bio_labels)


def add_text_snippet(span: Dict[str, object], text: str) -> Dict[str, object]:
    span_copy = dict(span)
    text_length = len(text)
    start = int(span_copy.get("start", 0))
    end = int(span_copy.get("end", start))
    start = max(0, min(text_length, start))
    end = max(start, min(text_length, end))
    span_copy["start"] = start
    span_copy["end"] = end
    span_copy["text"] = text[start:end]
    return span_copy


def classify_span_sets(predicted_spans, gold_spans):
    def key(span):
        return (span["start"], span["end"], span["label"])

    predicted_map = {key(span): span for span in predicted_spans}
    gold_map = {key(span): span for span in gold_spans}
    true_positive_keys = set(predicted_map) & set(gold_map)
    false_positive_keys = set(predicted_map) - set(gold_map)
    false_negative_keys = set(gold_map) - set(predicted_map)

    def sort_spans(keys, mapping):
        return [mapping[item] for item in sorted(keys, key=lambda row: (row[0], row[1], row[2]))]

    return (
        sort_spans(true_positive_keys, predicted_map),
        sort_spans(false_positive_keys, predicted_map),
        sort_spans(false_negative_keys, gold_map),
    )


def save_test_predictions(
    trainer,
    tokenized_dataset,
    raw_dataset,
    output_dir: str | Path,
    schema: LabelSchema,
    metadata_fields: Sequence[str] = (),
):
    import numpy as np

    prediction_output = trainer.predict(tokenized_dataset)
    predicted_ids = np.argmax(prediction_output.predictions, axis=2)
    label_ids = prediction_output.label_ids
    tokenized_python = tokenized_dataset.with_format("python")
    raw_python = raw_dataset.with_format("python")
    offsets_all = tokenized_python["offset_mapping"]
    detailed_records = []
    flat_rows = []
    metadata_fields = list(metadata_fields)

    for index in range(len(raw_python)):
        raw_example = raw_python[index]
        offsets = offsets_all[index]
        predicted_sequence = predicted_ids[index]
        label_sequence = label_ids[index]
        filtered_labels = []
        filtered_offsets = []
        for predicted_label_id, label_id, offset in zip(predicted_sequence, label_sequence, offsets):
            if label_id == -100:
                continue
            if isinstance(offset, dict):
                start = int(offset.get("start", 0))
                end = int(offset.get("end", 0))
            else:
                start = int(offset[0])
                end = int(offset[1])
            filtered_offsets.append((start, end))
            filtered_labels.append(schema.id_to_label.get(int(predicted_label_id), "O"))

        predicted_spans = bio_to_spans(filtered_labels, filtered_offsets)
        text = raw_example.get("text", "") or ""
        predicted_spans = [add_text_snippet(span, text) for span in predicted_spans]

        gold_spans = []
        for span in raw_example.get("spans", []) or []:
            cleaned = {"start": int(span.get("start", 0)), "end": int(span.get("end", 0)), "label": span.get("label")}
            gold_spans.append(add_text_snippet(cleaned, text))

        true_positive_spans, false_positive_spans, false_negative_spans = classify_span_sets(predicted_spans, gold_spans)
        metadata = {field: raw_example.get(field) for field in metadata_fields}
        detailed_records.append(
            {
                "index": index,
                "metadata": metadata,
                "text": text,
                "predicted_spans": predicted_spans,
                "gold_spans": gold_spans,
                "true_positives": true_positive_spans,
                "false_positives": false_positive_spans,
                "false_negatives": false_negative_spans,
            }
        )

        for category, spans in (("TP", true_positive_spans), ("FP", false_positive_spans), ("FN", false_negative_spans)):
            for span in spans:
                row = {
                    "example_index": index,
                    "category": category,
                    "label": span.get("label"),
                    "start": span.get("start"),
                    "end": span.get("end"),
                    "text_span": span.get("text", ""),
                }
                row.update(metadata)
                flat_rows.append(row)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "test_predictions_detailed.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in detailed_records:
            handle.write(json.dumps(record) + "\n")

    csv_path = output_path / "test_span_classification.csv"
    csv_fieldnames = ["example_index", "category", "label", "start", "end", "text_span"] + metadata_fields
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    return jsonl_path, csv_path


def evaluate_with_nervaluate(
    trainer,
    tokenized_dataset,
    records: Sequence[Dict[str, object]],
    id_to_label: Dict[int, str],
    label_to_id: Dict[str, int],
    output_dir: Path,
    suffix: str,
):
    import numpy as np
    from nervaluate import Evaluator

    prediction_output = trainer.predict(tokenized_dataset)
    metrics = {key: float(value) for key, value in prediction_output.metrics.items()}

    predicted_label_ids = np.argmax(prediction_output.predictions, axis=2)
    true_label_ids = prediction_output.label_ids

    predicted_sequences: List[List[str]] = []
    true_sequences: List[List[str]] = []
    for predicted_ids, true_ids in zip(predicted_label_ids, true_label_ids):
        sample_predictions: List[str] = []
        sample_truth: List[str] = []
        for predicted_id, true_id in zip(predicted_ids, true_ids):
            if true_id == -100:
                continue
            sample_predictions.append(id_to_label[int(predicted_id)])
            sample_truth.append(id_to_label[int(true_id)])
        predicted_sequences.append(sample_predictions)
        true_sequences.append(sample_truth)

    if not (len(predicted_sequences) == len(records) == len(true_sequences)):
        raise ValueError("Mismatch between prediction outputs and provided records for nervaluate evaluation.")

    gold_entities = []
    predicted_entities = []
    for record, gold_labels, predicted_labels in zip(records, true_sequences, predicted_sequences):
        token_spans = record.get("token_spans", [])
        gold_entities.append(bio_to_entities(token_spans, gold_labels))
        predicted_entities.append(bio_to_entities(token_spans, predicted_labels))

    entity_tags = sorted(
        {
            label.split("-", 1)[1]
            for label in label_to_id.keys()
            if label != "O" and "-" in label
        }
    )

    evaluator = Evaluator(gold_entities, predicted_entities, tags=entity_tags, loader="dict")
    evaluated = evaluator.evaluate()
    summary_lines = evaluator.summary_report().strip().splitlines()

    per_tag_sections: Dict[str, List[str]] = {}
    per_tag_lines: List[str] = []
    for scenario in NERVALUATE_SCENARIOS:
        scenario_report = evaluator.summary_report(mode="entities", scenario=scenario)
        lines = scenario_report.strip().splitlines()
        per_tag_sections[scenario] = lines
        if per_tag_lines:
            per_tag_lines.append("")
        per_tag_lines.extend(lines)

    gold_support = Counter(entity["label"] for document in gold_entities for entity in document)
    per_tag_results: Dict[str, Dict[str, object]] = {}
    for label in entity_tags:
        scenario_results: Dict[str, object] = {}
        raw_label_results = evaluated["entities"].get(label, {})
        for scenario in NERVALUATE_SCENARIOS:
            raw_result = raw_label_results.get(scenario)
            scenario_results[scenario] = evaluation_result_to_dict(raw_result) if raw_result is not None else None
        per_tag_results[label] = {
            "gold_support": int(gold_support.get(label, 0)),
            "evaluable": gold_support.get(label, 0) > 0,
            "scenarios": scenario_results,
        }

    evaluable_labels = [label for label in entity_tags if gold_support.get(label, 0) > 0]
    rollups: Dict[str, Dict[str, object]] = {}
    for scenario in NERVALUATE_SCENARIOS:
        overall_raw_result = evaluated["overall"].get(scenario)
        if overall_raw_result is None:
            continue
        label_results = [
            per_tag_results[label]["scenarios"][scenario]
            for label in evaluable_labels
            if per_tag_results[label]["scenarios"][scenario] is not None
        ]
        metrics_rollup = compute_micro_macro(
            scenario,
            evaluation_result_to_dict(overall_raw_result),
            label_results,
        )
        if metrics_rollup is not None:
            rollups[scenario] = {"overall": metrics_rollup}

    nervaluate_output = {
        "summary_lines": summary_lines,
        "per_tag_lines": per_tag_lines,
        "per_tag_sections": per_tag_sections,
        "labels": entity_tags,
        "per_tag_results": per_tag_results,
        "rollups": rollups,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"nervaluate_{suffix}.json", nervaluate_output)
    with (output_dir / f"nervaluate_{suffix}.txt").open("w", encoding="utf-8") as handle:
        handle.write("Summary (all scenarios)\n")
        handle.write("\n".join(summary_lines))
        handle.write("\n\nPer-entity breakdown\n")
        handle.write("\n".join(per_tag_lines))
        handle.write("\n")

    return {"trainer_metrics": metrics, "nervaluate": nervaluate_output}
