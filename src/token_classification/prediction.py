from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import TrainingConfig
from .data import load_prediction_dataset
from .labels import bio_to_spans
from .training import make_trainer
from .utils import ensure_directory, write_json

CHUNK_METADATA_FIELDS = {"chunk_index", "char_start", "char_end", "token_len", "annotation_count"}


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def infer_prediction_settings(model_path: str | Path, max_length: int | None, text_column: str | None):
    resolved_max_length = max_length if max_length is not None else 512
    resolved_text_column = text_column
    config_path = Path(model_path) / "config_used.yaml"
    if config_path.exists():
        training_config = TrainingConfig.from_yaml(config_path)
        if max_length is None:
            resolved_max_length = training_config.max_length
        if text_column is None:
            resolved_text_column = training_config.text_column
    return resolved_max_length, resolved_text_column


def _tokenize_for_prediction(dataset, tokenizer, max_length: int):
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )

    return dataset.map(tokenize, batched=False)


def _integer_metadata(metadata: dict[str, Any], field: str, default: int = 0) -> int:
    value = metadata.get(field, default)
    if value in (None, ""):
        return default
    return int(value)


def aggregate_chunk_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore document-relative offsets and deduplicate exact predictions from overlapping chunks."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    for record in records:
        metadata = record.get("metadata", {}) or {}
        document_id = metadata.get("document_id") or metadata.get("doc_key") or metadata.get("source_row_id")
        group_key = str(document_id) if document_id not in (None, "") else f"row:{record.get('index', len(group_order))}"
        if group_key not in grouped:
            grouped[group_key] = []
            group_order.append(group_key)
        grouped[group_key].append(record)

    documents = []
    for group_key in group_order:
        chunks = sorted(
            grouped[group_key],
            key=lambda row: (
                _integer_metadata(row.get("metadata", {}) or {}, "char_start"),
                _integer_metadata(row.get("metadata", {}) or {}, "chunk_index"),
            ),
        )
        maximum_end = max(
            (
                _integer_metadata(chunk.get("metadata", {}) or {}, "char_end", len(chunk.get("text", "")))
                for chunk in chunks
            ),
            default=0,
        )
        characters = [""] * maximum_end
        span_map: dict[tuple[int, int, str], dict[str, Any]] = {}
        for chunk in chunks:
            metadata = chunk.get("metadata", {}) or {}
            char_start = _integer_metadata(metadata, "char_start")
            chunk_text = chunk.get("text", "") or ""
            required_length = char_start + len(chunk_text)
            if required_length > len(characters):
                characters.extend([""] * (required_length - len(characters)))
            for offset, character in enumerate(chunk_text):
                characters[char_start + offset] = character
            for span in chunk.get("predicted_spans", []):
                global_span = dict(span)
                global_span["start"] = int(span["start"]) + char_start
                global_span["end"] = int(span["end"]) + char_start
                key = (global_span["start"], global_span["end"], str(global_span["label"]))
                span_map.setdefault(key, global_span)

        text = "".join(character or " " for character in characters)
        predicted_spans = sorted(span_map.values(), key=lambda span: (span["start"], span["end"], span["label"]))
        for span in predicted_spans:
            span["text"] = text[span["start"] : span["end"]]
        first_metadata = chunks[0].get("metadata", {}) or {}
        document_metadata = {
            field: value for field, value in first_metadata.items() if field not in CHUNK_METADATA_FIELDS
        }
        documents.append(
            {
                "document_id": group_key,
                "metadata": document_metadata,
                "text": text,
                "predicted_spans": predicted_spans,
            }
        )
    return documents


def _save_document_predictions(records: list[dict[str, Any]], output_path: Path) -> tuple[Path, Path]:
    jsonl_path = output_path / "document_predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = output_path / "document_predicted_spans.csv"
    fieldnames = ["document_id", "label", "start", "end", "text_span"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for span in record["predicted_spans"]:
                writer.writerow(
                    {
                        "document_id": record["document_id"],
                        "label": span["label"],
                        "start": span["start"],
                        "end": span["end"],
                        "text_span": span.get("text", ""),
                    }
                )
    return jsonl_path, csv_path


def save_prediction_outputs(trainer, tokenized_dataset, raw_dataset, output_dir: str | Path, id_to_label: dict[int, str]):
    import numpy as np

    prediction_output = trainer.predict(tokenized_dataset)
    predicted_ids = np.argmax(prediction_output.predictions, axis=2)
    tokenized_python = tokenized_dataset.with_format("python")
    raw_python = raw_dataset.with_format("python")
    offsets_all = tokenized_python["offset_mapping"]
    metadata_fields = [column for column in raw_dataset.column_names if column != "text"]

    detailed_records = []
    flat_rows = []
    label_counts: dict[str, int] = {}
    rows_with_predictions = 0
    total_predicted_spans = 0

    for index in range(len(raw_python)):
        raw_example = raw_python[index]
        offsets = offsets_all[index]
        predicted_sequence = predicted_ids[index]
        filtered_labels = []
        filtered_offsets = []
        for predicted_label_id, offset in zip(predicted_sequence, offsets):
            if isinstance(offset, dict):
                start = int(offset.get("start", 0))
                end = int(offset.get("end", 0))
            else:
                start = int(offset[0])
                end = int(offset[1])
            if start == 0 and end == 0:
                continue
            filtered_offsets.append((start, end))
            filtered_labels.append(id_to_label.get(int(predicted_label_id), "O"))

        predicted_spans = bio_to_spans(filtered_labels, filtered_offsets)
        text = raw_example.get("text", "") or ""
        metadata = {field: raw_example.get(field) for field in metadata_fields}
        if predicted_spans:
            rows_with_predictions += 1
        total_predicted_spans += len(predicted_spans)

        for span in predicted_spans:
            span["text"] = text[span["start"] : span["end"]]
            label = str(span["label"])
            label_counts[label] = label_counts.get(label, 0) + 1
            row = {
                "example_index": index,
                "label": label,
                "start": span["start"],
                "end": span["end"],
                "text_span": span["text"],
            }
            row.update({field: _serialize_value(metadata[field]) for field in metadata_fields})
            flat_rows.append(row)

        detailed_records.append(
            {
                "index": index,
                "metadata": metadata,
                "text": text,
                "predicted_spans": predicted_spans,
            }
        )

    output_path = ensure_directory(output_dir)
    jsonl_path = output_path / "predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in detailed_records:
            handle.write(json.dumps(record) + "\n")

    csv_path = output_path / "predicted_spans.csv"
    csv_fieldnames = ["example_index", "label", "start", "end", "text_span"] + metadata_fields
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    document_records = aggregate_chunk_predictions(detailed_records)
    document_jsonl_path, document_csv_path = _save_document_predictions(document_records, output_path)

    summary = {
        "row_count": len(raw_python),
        "document_count": len(document_records),
        "rows_with_predictions": rows_with_predictions,
        "total_predicted_spans": total_predicted_spans,
        "label_counts": label_counts,
        "output_files": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "document_jsonl": str(document_jsonl_path),
            "document_csv": str(document_csv_path),
        },
    }
    summary_path = write_json(output_path / "prediction_summary.json", summary)
    return summary_path, jsonl_path, csv_path


def run_prediction(
    *,
    model_path: str | Path,
    input_file: str | Path,
    output_dir: str | Path | None = None,
    text_column: str | None = None,
    max_length: int | None = None,
    batch_size: int = 8,
    dataloader_num_workers: int = 0,
) -> Path:
    from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

    model_path = Path(model_path)
    input_file = Path(input_file)
    if output_dir is None:
        output_dir = Path("outputs") / "predictions" / f"{model_path.name}_{input_file.stem}"
    output_dir = ensure_directory(output_dir)

    resolved_max_length, resolved_text_column = infer_prediction_settings(model_path, max_length=max_length, text_column=text_column)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    dataset = load_prediction_dataset(str(input_file), text_column=resolved_text_column)
    tokenized_dataset = _tokenize_for_prediction(dataset, tokenizer, max_length=resolved_max_length)

    model = AutoModelForTokenClassification.from_pretrained(str(model_path))
    id_to_label = {int(key): value for key, value in model.config.id2label.items()}

    inference_args = make_training_args_for_prediction(
        output_dir=output_dir,
        batch_size=batch_size,
        dataloader_num_workers=dataloader_num_workers,
    )
    trainer = make_trainer(
        model=model,
        args=inference_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    summary_path, jsonl_path, csv_path = save_prediction_outputs(
        trainer,
        tokenized_dataset,
        dataset,
        output_dir,
        id_to_label=id_to_label,
    )
    print("\nPrediction outputs written to:")
    print(f"  Summary: {summary_path}")
    print(f"  JSONL:   {jsonl_path}")
    print(f"  CSV:     {csv_path}")
    return output_dir


def make_training_args_for_prediction(*, output_dir: str | Path, batch_size: int, dataloader_num_workers: int):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=dataloader_num_workers,
        report_to="none",
    )
