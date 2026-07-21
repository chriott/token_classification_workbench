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

    summary = {
        "row_count": len(raw_python),
        "rows_with_predictions": rows_with_predictions,
        "total_predicted_spans": total_predicted_spans,
        "label_counts": label_counts,
        "output_files": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
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
