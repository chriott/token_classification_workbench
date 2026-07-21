import ast
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from nervaluate import Evaluator
from datasets import DatasetDict, load_dataset, Value
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

BEST_MODEL_METRIC = os.environ.get("PRIMARY_METRIC", "eval_f1_macro")
REPORT_TARGET = "none"

def _parse_bool_env(env_name: str, default: bool) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TrainingConfig:
    model_name: str = os.environ.get("MODEL_NAME", "roberta-large")
    output_dir: str = os.environ.get("OUTPUT_DIR", "./roberta-ipi-ft-train_synth_phi_ogtest")
    run_name: str = os.environ.get("RUN_NAME", os.environ.get("FINAL_RUN_NAME", "final_lr5e-5_bs16_ep32"))
    max_length: int = int(os.environ.get("MAX_LENGTH", "512"))
    split_seed: int = int(os.environ.get("SPLIT_SEED", "42"))
    train_learning_rate: float = float(os.environ.get("TRAIN_LEARNING_RATE", "5e-5"))
    train_batch_size: int = int(os.environ.get("TRAIN_BATCH_SIZE", "16"))
    train_epochs: float = float(os.environ.get("TRAIN_EPOCHS", "30"))
    train_weight_decay: float = float(os.environ.get("TRAIN_WEIGHT_DECAY", "0.01"))
    text_column: Optional[str] = os.environ.get("TEXT_COLUMN")
    spans_column: str = os.environ.get("SPANS_COLUMN", "spans")
    optional_string_columns: Tuple[str, ...] = ("subject_id", "hadm_id", "chartdate")
    train_file: str = os.environ.get("TRAIN_FILE", "train_chunks_512_ibra_phi_synth.csv")
    validation_file: Optional[str] = os.environ.get("VALIDATION_FILE", "validation_chunks_512_ibra_phi_synth.csv")
    test_file: str = os.environ.get("TEST_FILE", "test_chunks_512_ibra.csv")


@dataclass
class LabelSchema:
    labels: List[str]
    bio_labels: List[str]
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]

    @classmethod
    def from_dataset(cls, dataset) -> "LabelSchema":
        labels = set()
        for example in dataset:
            for span in example.get("spans", []):
                label = span.get("label")
                if label:
                    labels.add(label)
        sorted_labels = sorted(labels)
        bio_labels = ["O"] + [f"{prefix}-{label}" for label in sorted_labels for prefix in ("B", "I")]
        label_to_id = {label: idx for idx, label in enumerate(bio_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        return cls(sorted_labels, bio_labels, label_to_id, id_to_label)


def _find_text_col(cols):
    lowered = {c.lower(): c for c in cols}
    for cand in ("text", "note_text", "document", "doc", "notes", "content"):
        if cand in lowered:
            return lowered[cand]
    for cand in ("TEXT", "Text"):
        if cand in cols:
            return cand
    return None


def _ensure_text_column(dataset, text_column: Optional[str]):
    if text_column and text_column in dataset.column_names:
        source_col = text_column
    else:
        source_col = "text" if "text" in dataset.column_names else _find_text_col(dataset.column_names)
        if source_col is None:
            raise ValueError(
                "CSV does not contain a 'text' column and no text source was found.\n"
                "Options:\n  • Add a 'text' column to your CSV."
            )
    if source_col != "text":
        dataset = dataset.rename_column(source_col, "text")
    return dataset


def _normalize_optional_string_columns(dataset, optional_columns: Sequence[str]):
    for column in optional_columns:
        if column in dataset.column_names:
            dataset = dataset.map(
                lambda example, col=column: {col: "" if example[col] is None else str(example[col])},
                desc=f"normalizing column {column}",
            )
            try:
                dataset = dataset.cast_column(column, Value("string"))
            except (TypeError, ValueError):
                pass
    return dataset


def _make_parse_spans_fn(spans_column: str):
    def _parse(example):
        spans_val = example.get(spans_column, None)
        if isinstance(spans_val, str):
            try:
                example["spans"] = ast.literal_eval(spans_val.strip())
            except Exception:
                cleaned = spans_val.replace("“", '"').replace("”", '"').replace("’", "'")
                try:
                    example["spans"] = ast.literal_eval(cleaned)
                except Exception:
                    example["spans"] = []
        elif isinstance(spans_val, list):
            example["spans"] = spans_val
        else:
            example["spans"] = []
        return example

    return _parse


def load_and_prepare_dataset(path, config: TrainingConfig):
    if isinstance(path, str):
        path_parts = [part.strip() for part in path.split(",") if part.strip()]
    elif isinstance(path, Sequence):
        path_parts = list(path)
    else:
        path_parts = [path]
    data_files = {"data": path_parts if len(path_parts) > 1 else path_parts[0]}
    loaded = load_dataset("csv", data_files=data_files)["data"]
    loaded = _ensure_text_column(loaded, config.text_column)
    loaded = loaded.map(_make_parse_spans_fn(config.spans_column))
    loaded = _normalize_optional_string_columns(loaded, config.optional_string_columns)
    return loaded


def load_dataset_splits(config: TrainingConfig) -> DatasetDict:
    ds = DatasetDict()
    ds["train"] = load_and_prepare_dataset(config.train_file, config)
    if config.validation_file:
        ds["validation"] = load_and_prepare_dataset(config.validation_file, config)
    ds["test"] = load_and_prepare_dataset(config.test_file, config)
    return ds


def _build_label_aligner(tokenizer, schema: LabelSchema, max_length: int):
    def align_labels_with_tokens(example):
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )
        labels = []
        for (start, end) in tokenized["offset_mapping"]:
            labels.append(-100 if start == 0 and end == 0 else schema.label_to_id["O"])

        unmapped = []
        for span in example.get("spans", []):
            start_char = span.get("start")
            end_char = span.get("end")
            label = span.get("label")
            if label not in schema.labels or start_char is None or end_char is None:
                continue
            token_start = tokenized.char_to_token(start_char)
            token_end = tokenized.char_to_token(end_char - 1)
            if token_start is None or token_end is None:
                unmapped.append(span)
                continue
            if labels[token_start] != -100:
                labels[token_start] = schema.label_to_id[f"B-{label}"]
            for idx in range(token_start + 1, token_end + 1):
                if labels[idx] != -100:
                    labels[idx] = schema.label_to_id[f"I-{label}"]
        if unmapped:
            print(
                f"Warning: could not align {len(unmapped)} spans in example {example.get('id', 'unknown')}: {unmapped}"
            )
        tokenized["labels"] = labels
        return tokenized

    return align_labels_with_tokens


def _build_compute_metrics_fn(schema: LabelSchema):
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = [[schema.id_to_label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [schema.id_to_label[pred] for pred, label in zip(pred_row, label_row) if label != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]

        p_micro = precision_score(true_labels, true_predictions, average="micro")
        r_micro = recall_score(true_labels, true_predictions, average="micro")
        f1_micro = f1_score(true_labels, true_predictions, average="micro")
        p_macro = precision_score(true_labels, true_predictions, average="macro")
        r_macro = recall_score(true_labels, true_predictions, average="macro")
        f1_macro = f1_score(true_labels, true_predictions, average="macro")
        return {
            "eval_accuracy": accuracy_score(true_labels, true_predictions),
            "precision_micro": p_micro,
            "recall_micro": r_micro,
            "f1_micro": f1_micro,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f1_macro,
        }

    return compute_metrics


def instantiate_model(config: TrainingConfig, schema: LabelSchema):
    return AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(schema.bio_labels),
        id2label=schema.id_to_label,
        label2id=schema.label_to_id,
    )


def _persist_label_schema(schema: LabelSchema, output_dir: str) -> None:
    with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(schema.label_to_id, f, indent=2)
    id2label_str_keys = {str(k): v for k, v in schema.id_to_label.items()}
    with open(os.path.join(output_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label_str_keys, f, indent=2)
    with open(os.path.join(output_dir, "bio_labels.json"), "w", encoding="utf-8") as f:
        json.dump(schema.bio_labels, f, indent=2)


def _bio_to_spans(labels, offsets):
    spans = []
    current_label = None
    current_start = None
    current_end = None
    for label, offset in zip(labels, offsets):
        start, end = offset
        if start == 0 and end == 0:
            continue
        if label in (None, "O"):
            if current_label is not None:
                spans.append({"start": current_start, "end": current_end, "label": current_label})
                current_label = None
            continue
        if "-" in label:
            prefix, entity_type = label.split("-", 1)
        else:
            prefix, entity_type = "B", label
        if prefix == "B" or entity_type != current_label:
            if current_label is not None:
                spans.append({"start": current_start, "end": current_end, "label": current_label})
            current_label = entity_type
            current_start = start
            current_end = end
        else:
            current_end = end
    if current_label is not None:
        spans.append({"start": current_start, "end": current_end, "label": current_label})
    return spans


def _add_text_snippet(span, text):
    span_copy = dict(span)
    text_len = len(text)
    start = int(span_copy.get("start", 0))
    end = int(span_copy.get("end", start))
    start = max(0, min(text_len, start))
    end = max(start, min(text_len, end))
    span_copy["start"] = start
    span_copy["end"] = end
    span_copy["text"] = text[start:end]
    return span_copy

#helper because nervaluate only accepts span lists rather than bio tags
def bio_to_entities(token_spans: Sequence[Sequence[int]], labels: Sequence[str]) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    current_label: str = ""
    start_char: int = -1
    end_char: int = -1

    for idx, label in enumerate(labels):
        if idx >= len(token_spans):
            break
        if label == "O":
            if current_label:
                entities.append({"label": current_label, "start": start_char, "end": end_char})
                current_label = ""
                start_char = -1
                end_char = -1
            continue
        if "-" in label:
            prefix, entity_type = label.split("-", 1)
        else:
            prefix, entity_type = "B", label
        span_start, span_end = token_spans[idx]
        if prefix == "B" or not current_label or entity_type != current_label:
            if current_label:
                entities.append({"label": current_label, "start": start_char, "end": end_char})
            current_label = entity_type
            start_char = span_start
            end_char = span_end
        else:
            current_label = entity_type
            end_char = span_end

    if current_label:
        entities.append({"label": current_label, "start": start_char, "end": end_char})

    return entities


def _classify_span_sets(predicted_spans, gold_spans):
    def _key(span):
        return (span["start"], span["end"], span["label"])

    pred_map = {_key(span): span for span in predicted_spans}
    gold_map = {_key(span): span for span in gold_spans}
    tp_keys = set(pred_map.keys()) & set(gold_map.keys())
    fp_keys = set(pred_map.keys()) - set(gold_map.keys())
    fn_keys = set(gold_map.keys()) - set(pred_map.keys())

    def _sorted(keys, mapping):
        return [mapping[k] for k in sorted(keys, key=lambda item: (item[0], item[1], item[2]))]

    return _sorted(tp_keys, pred_map), _sorted(fp_keys, pred_map), _sorted(fn_keys, gold_map)


def _save_test_predictions(trainer, tokenized_dataset, raw_dataset, output_dir: str, schema: LabelSchema):
    prediction_output = trainer.predict(tokenized_dataset)
    pred_ids = np.argmax(prediction_output.predictions, axis=2)
    label_ids = prediction_output.label_ids
    tokenized_python = tokenized_dataset.with_format("python")
    raw_python = raw_dataset.with_format("python")
    offsets_all = tokenized_python["offset_mapping"]
    detailed_records = []
    flat_rows = []
    metadata_fields = ["source_row_id", "chunk_index", "subject_id", "hadm_id", "chartdate"]

    for idx in range(len(raw_python)):
        raw_example = raw_python[idx]
        offsets = offsets_all[idx]
        pred_seq = pred_ids[idx]
        label_seq = label_ids[idx]
        filtered_labels = []
        filtered_offsets = []
        for pred_label_id, label_id, offset in zip(pred_seq, label_seq, offsets):
            if label_id == -100:
                continue
            if isinstance(offset, dict):
                start = int(offset.get("start", 0))
                end = int(offset.get("end", 0))
            else:
                start = int(offset[0])
                end = int(offset[1])
            filtered_offsets.append((start, end))
            filtered_labels.append(schema.id_to_label.get(int(pred_label_id), "O"))
        predicted_spans = _bio_to_spans(filtered_labels, filtered_offsets)
        text = raw_example.get("text", "") or ""
        predicted_spans = [_add_text_snippet(span, text) for span in predicted_spans]

        gold_spans_raw = raw_example.get("spans", []) or []
        gold_spans = []
        for span in gold_spans_raw:
            cleaned = {"start": int(span.get("start", 0)), "end": int(span.get("end", 0)), "label": span.get("label")}
            gold_spans.append(_add_text_snippet(cleaned, text))

        tp_spans, fp_spans, fn_spans = _classify_span_sets(predicted_spans, gold_spans)
        metadata = {field: raw_example.get(field) for field in metadata_fields}

        detailed_records.append(
            {
                "index": idx,
                "metadata": metadata,
                "text": text,
                "predicted_spans": predicted_spans,
                "gold_spans": gold_spans,
                "true_positives": tp_spans,
                "false_positives": fp_spans,
                "false_negatives": fn_spans,
            }
        )

        for category, spans in (("TP", tp_spans), ("FP", fp_spans), ("FN", fn_spans)):
            for span in spans:
                row = {
                    "example_index": idx,
                    "category": category,
                    "label": span.get("label"),
                    "start": span.get("start"),
                    "end": span.get("end"),
                    "text_span": span.get("text", ""),
                }
                row.update(metadata)
                flat_rows.append(row)

    jsonl_path = os.path.join(output_dir, "test_predictions_detailed.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for record in detailed_records:
            jf.write(json.dumps(record) + "\n")

    csv_fieldnames = ["example_index", "category", "label", "start", "end", "text_span"] + metadata_fields
    csv_path = os.path.join(output_dir, "test_span_classification.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    return jsonl_path, csv_path


def evaluate_with_nervaluate(
    trainer,
    tokenized_dataset,
    records: Sequence[Dict[str, object]],
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    output_dir: Path,
    suffix: str,
):

    prediction_output = trainer.predict(tokenized_dataset)
    metrics = {key: float(value) for key, value in prediction_output.metrics.items()}

    predicted_label_ids = np.argmax(prediction_output.predictions, axis=2)
    true_label_ids = prediction_output.label_ids

    predicted_sequences: List[List[str]] = []
    true_sequences: List[List[str]] = []

    for pred_ids, true_ids in zip(predicted_label_ids, true_label_ids):
        sample_preds: List[str] = []
        sample_true: List[str] = []
        for pred_id, true_id in zip(pred_ids, true_ids):
            if true_id == -100:
                continue
            sample_preds.append(id2label[int(pred_id)])
            sample_true.append(id2label[int(true_id)])
        predicted_sequences.append(sample_preds)
        true_sequences.append(sample_true)

    if not (
        len(predicted_sequences) == len(records) == len(true_sequences)
    ):
        raise ValueError("Mismatch between prediction outputs and provided records for nervaluate evaluation.")

    gold_entities = []
    pred_entities = []
    for record, gold_labels, pred_labels in zip(records, true_sequences, predicted_sequences):
        token_spans = record.get("token_spans", [])
        gold_entities.append(bio_to_entities(token_spans, gold_labels))
        pred_entities.append(bio_to_entities(token_spans, pred_labels))

    entity_tags = sorted(
        {
            label.split("-", 1)[1]
            for label in label2id.keys()
            if label != "O" and "-" in label
        }
    )

    evaluator = Evaluator(gold_entities, pred_entities, tags=entity_tags, loader="dict")
    summary_lines = evaluator.summary_report().strip().splitlines()

    per_tag_sections: Dict[str, List[str]] = {}
    per_tag_lines: List[str] = []
    for scenario in ("strict", "ent_type", "partial"):
        scenario_report = evaluator.summary_report(mode="entities", scenario=scenario)
        lines = scenario_report.strip().splitlines()
        per_tag_sections[scenario] = lines
        if per_tag_lines:
            per_tag_lines.append("")
        per_tag_lines.extend(lines)

    def _parse_per_tag_rows(lines: List[str]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            label = parts[0]
            try:
                correct, incorrect, partial, missed, spurious = [int(parts[i]) for i in range(1, 6)]
                precision, recall, f1 = [float(parts[i]) for i in range(6, 9)]
            except ValueError:
                continue
            rows.append(
                {
                    "label": label,
                    "correct": correct,
                    "incorrect": incorrect,
                    "partial": partial,
                    "missed": missed,
                    "spurious": spurious,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        return rows

    def _compute_micro_macro(rows: List[Dict[str, object]]):
        if not rows:
            return None
        totals = {key: 0 for key in ("correct", "missed", "spurious")}
        for row in rows:
            totals["correct"] += row["correct"]
            totals["missed"] += row["missed"]
            totals["spurious"] += row["spurious"]
        tp = totals["correct"]
        fp = totals["spurious"]
        fn = totals["missed"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        macro_f1 = sum(row["f1"] for row in rows) / len(rows)
        return {
            "micro": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            },
            "macro": {
                "f1": macro_f1,
                "label_count": len(rows),
            },
        }

    rollups: Dict[str, Dict[str, object]] = {}
    for scenario in ("ent_type", "partial"):
        rows = _parse_per_tag_rows(per_tag_sections.get(scenario, []))
        if not rows:
            continue
        groups = {
            "overall": rows,
            "phi": [row for row in rows if row["label"].startswith("PHI-")],
            "ipi": [row for row in rows if row["label"].startswith("IPI-")],
        }
        scenario_rollups: Dict[str, object] = {}
        for group_name, group_rows in groups.items():
            metrics_rollup = _compute_micro_macro(group_rows)
            if metrics_rollup is not None:
                scenario_rollups[group_name] = metrics_rollup
        if scenario_rollups:
            rollups[scenario] = scenario_rollups

    nervaluate_output = {
        "summary_lines": summary_lines,
        "per_tag_lines": per_tag_lines,
        "per_tag_sections": per_tag_sections,
        "rollups": rollups,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"nervaluate_{suffix}.json"
    txt_path = output_dir / f"nervaluate_{suffix}.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(nervaluate_output, handle, indent=2)

    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("Summary (all scenarios)\n")
        handle.write("\n".join(summary_lines))
        handle.write("\n\nPer-entity breakdown\n")
        handle.write("\n".join(per_tag_lines))
        handle.write("\n")

    return {"trainer_metrics": metrics, "nervaluate": nervaluate_output}


def make_training_args(config: TrainingConfig, run_output_dir: str, evaluation_enabled: bool):
    per_device_train_bs = config.train_batch_size
    per_device_eval_bs = config.train_batch_size
    minimal = dict(
        output_dir=run_output_dir,
        per_device_train_batch_size=per_device_train_bs,
        per_device_eval_batch_size=per_device_eval_bs,
        num_train_epochs=config.train_epochs,
        learning_rate=config.train_learning_rate,
        weight_decay=config.train_weight_decay,
        seed=config.split_seed,
    )
    common = dict(
        **minimal,
        logging_dir=os.path.join(run_output_dir, "logs"),
        logging_steps=int(os.environ.get("LOGGING_STEPS", "200")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "1")),
        fp16=_parse_bool_env("FP16", True),
        save_total_limit=int(os.environ.get("SAVE_TOTAL_LIMIT", "1")),
        run_name=config.run_name,
    )
    common_extra = dict(
        warmup_ratio=float(os.environ.get("WARMUP_RATIO", "0.06")),
        dataloader_num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
        report_to=REPORT_TARGET,
        evaluation_strategy="epoch" if evaluation_enabled else "no",
        save_strategy="epoch" if evaluation_enabled else "epoch",
    )
    if evaluation_enabled:
        common_extra.update(
            load_best_model_at_end=True,
            metric_for_best_model=BEST_MODEL_METRIC,
            greater_is_better=True,
        )
    try:
        return TrainingArguments(
            **common,
            **common_extra,
        )
    except TypeError:
        return TrainingArguments(**minimal)


def run_pipeline():
    config = TrainingConfig()
    ds = load_dataset_splits(config)
    if "validation" not in ds:
        raise ValueError("Validation split is required for training. Set VALIDATION_FILE to a valid CSV.")

    schema = LabelSchema.from_dataset(ds["train"])
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    align_fn = _build_label_aligner(tokenizer, schema, config.max_length)
    tokenized_ds = ds.map(align_fn, batched=False)
    train_dataset = tokenized_ds["train"]
    validation_dataset = tokenized_ds["validation"]
    compute_metrics_fn = _build_compute_metrics_fn(schema)

    os.makedirs(config.output_dir, exist_ok=True)
    run_output_dir = os.path.join(config.output_dir, config.run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    training_args = make_training_args(config, run_output_dir, evaluation_enabled=len(validation_dataset) > 0)
    model = instantiate_model(config, schema)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics_fn,
    )

    print(f"\n=== Training run: {config.run_name} ===")
    trainer.train()
    trainer.save_model(run_output_dir)
    tokenizer.save_pretrained(run_output_dir)

    eval_metrics = {k: float(v) for k, v in trainer.evaluate(validation_dataset).items()}
    print("\nValidation metrics:")
    print(json.dumps(eval_metrics, indent=2))

    best_run = {
        "run_name": config.run_name,
        "output_dir": run_output_dir,
        "params": {
            "learning_rate": config.train_learning_rate,
            "per_device_train_batch_size": config.train_batch_size,
            "per_device_eval_batch_size": config.train_batch_size,
            "num_train_epochs": config.train_epochs,
            "weight_decay": config.train_weight_decay,
            "seed": config.split_seed,
        },
        "metrics": eval_metrics,
    }

    print("\nTraining completed. Saved model to", run_output_dir)
    print("Best run hyperparameters:")
    print(json.dumps(best_run["params"], indent=2))

    test_dataset = tokenized_ds["test"]
    test_args = TrainingArguments(
        output_dir=os.path.join(run_output_dir, "test_eval"),
        per_device_eval_batch_size=config.train_batch_size,
        dataloader_num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
        report_to="none",
    )
    best_model = AutoModelForTokenClassification.from_pretrained(run_output_dir)
    best_trainer = Trainer(
        model=best_model,
        args=test_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics_fn,
    )
    test_metrics = {k: float(v) for k, v in best_trainer.evaluate(test_dataset).items()}
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))

    with open(os.path.join(run_output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    _persist_label_schema(schema, run_output_dir)
    pred_jsonl_path, pred_csv_path = _save_test_predictions(
        best_trainer,
        test_dataset,
        ds["test"],
        run_output_dir,
        schema,
    )
    print(f"\nSaved detailed prediction outputs to:\n  JSONL: {pred_jsonl_path}\n  CSV:   {pred_csv_path}")

    tokenized_test_python = tokenized_ds["test"].with_format("python")
    test_records = []
    for offsets in tokenized_test_python["offset_mapping"]:
        spans = []
        for offset in offsets:
            if isinstance(offset, dict):
                start = int(offset.get("start", 0))
                end = int(offset.get("end", 0))
            else:
                start = int(offset[0])
                end = int(offset[1])
            if start == 0 and end == 0:
                continue
            spans.append((start, end))
        test_records.append({"token_spans": spans})

    nervaluate_dir = Path(run_output_dir) / "nervaluate"
    try:
        nervaluate_results = evaluate_with_nervaluate(
            best_trainer,
            test_dataset,
            test_records,
            schema.id_to_label,
            schema.label_to_id,
            nervaluate_dir,
            suffix="test",
        )
        print("\nNervaluate evaluation written to", nervaluate_dir)
        print(json.dumps(nervaluate_results["trainer_metrics"], indent=2))
    except SystemExit as exc:
        print("\nSkipping Nervaluate evaluation:", exc)

def main():
    run_pipeline()


if __name__ == "__main__":
    main()
