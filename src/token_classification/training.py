from __future__ import annotations

import inspect
import json
from pathlib import Path

from .config import TrainingConfig, dump_config_yaml
from .data import load_dataset_splits
from .evaluation import (
    build_compute_metrics_fn,
    evaluate_with_nervaluate,
    persist_label_schema,
    save_test_predictions,
)
from .labels import LabelSchema, build_label_aligner
from .utils import ensure_directory, write_json


def instantiate_model(config: TrainingConfig, schema: LabelSchema):
    from transformers import AutoModelForTokenClassification

    return AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(schema.bio_labels),
        id2label=schema.id_to_label,
        label2id=schema.label_to_id,
    )


def seed_training_run(seed: int) -> None:
    from transformers import set_seed

    set_seed(seed)


def make_training_args(config: TrainingConfig, run_output_dir: str | Path, evaluation_enabled: bool):
    from transformers import TrainingArguments

    run_output_dir = str(run_output_dir)
    minimal = dict(
        output_dir=run_output_dir,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.train_batch_size,
        num_train_epochs=config.train_epochs,
        learning_rate=config.train_learning_rate,
        weight_decay=config.train_weight_decay,
        seed=config.split_seed,
    )
    common = dict(
        **minimal,
        logging_dir=str(Path(run_output_dir) / "logs"),
        logging_steps=config.logging_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        save_total_limit=config.save_total_limit,
        run_name=config.run_name,
    )
    common_extra = dict(
        warmup_ratio=config.warmup_ratio,
        dataloader_num_workers=config.dataloader_num_workers,
        report_to=config.report_target,
        evaluation_strategy="epoch" if evaluation_enabled else "no",
        save_strategy="epoch",
    )
    if evaluation_enabled:
        common_extra.update(
            load_best_model_at_end=True,
            metric_for_best_model=config.primary_metric,
            greater_is_better=True,
        )
    requested_kwargs = {**common, **common_extra}
    supported_parameters = set(inspect.signature(TrainingArguments.__init__).parameters)
    compatibility_aliases = {
        "evaluation_strategy": ("evaluation_strategy", "eval_strategy"),
    }

    resolved_kwargs = {}
    unsupported_arguments = []
    for key, value in requested_kwargs.items():
        if key in compatibility_aliases:
            alias = next((candidate for candidate in compatibility_aliases[key] if candidate in supported_parameters), None)
            if alias is None:
                unsupported_arguments.append(key)
                continue
            resolved_kwargs[alias] = value
            continue
        if key in supported_parameters:
            resolved_kwargs[key] = value
            continue
        unsupported_arguments.append(key)

    if unsupported_arguments:
        unsupported_list = ", ".join(sorted(unsupported_arguments))
        raise RuntimeError(
            "Installed transformers.TrainingArguments does not support the required arguments: "
            f"{unsupported_list}. Install a compatible transformers version."
        )

    return TrainingArguments(**resolved_kwargs)


def make_trainer(*, tokenizer, **trainer_kwargs):
    from transformers import Trainer

    supported_parameters = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in supported_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError(
            "Installed transformers.Trainer does not support either 'processing_class' or 'tokenizer'. "
            "Install a compatible transformers version."
        )

    return Trainer(**trainer_kwargs)


def build_test_records(tokenized_test_dataset):
    tokenized_test_python = tokenized_test_dataset.with_format("python")
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
    return test_records


def merge_training_splits_for_final_mode(
    dataset_splits,
    *,
    final_training_mode: bool,
    concatenate_fn=None,
    dataset_dict_factory=None,
):
    if not final_training_mode or "validation" not in dataset_splits:
        return dataset_splits

    if concatenate_fn is None or dataset_dict_factory is None:
        from datasets import DatasetDict, concatenate_datasets

        concatenate_fn = concatenate_datasets
        dataset_dict_factory = DatasetDict

    merged_splits = dataset_dict_factory()
    merged_splits["train"] = concatenate_fn([dataset_splits["train"], dataset_splits["validation"]])
    for split_name, dataset in dataset_splits.items():
        if split_name == "validation":
            continue
        if split_name == "train":
            continue
        merged_splits[split_name] = dataset
    return merged_splits


def run_pipeline(
    config: TrainingConfig,
    run_test_evaluation: bool = True,
    final_training_mode: bool = False,
    save_model: bool = True,
    write_validation_nervaluate: bool = False,
) -> Path:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from transformers import DataCollatorForTokenClassification, EarlyStoppingCallback, TrainingArguments

    config.validate(require_validation=not final_training_mode)
    dataset_splits = load_dataset_splits(config)
    dataset_splits = merge_training_splits_for_final_mode(
        dataset_splits,
        final_training_mode=final_training_mode,
    )
    if not final_training_mode and "validation" not in dataset_splits:
        raise ValueError("Validation split is required for training. Set validation_file in the YAML config.")

    schema = LabelSchema.from_dataset(dataset_splits["train"])
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    align_fn = build_label_aligner(tokenizer, schema, config.max_length)
    tokenized_dataset = dataset_splits.map(align_fn, batched=False)

    train_dataset = tokenized_dataset["train"]
    validation_dataset = tokenized_dataset.get("validation")
    test_dataset = tokenized_dataset["test"]
    compute_metrics_fn = build_compute_metrics_fn(schema)

    ensure_directory(config.output_dir)
    run_output_dir = ensure_directory(Path(config.output_dir) / config.run_name)
    dump_config_yaml(run_output_dir / "config_used.yaml", config)

    evaluation_enabled = not final_training_mode and validation_dataset is not None and len(validation_dataset) > 0
    training_args = make_training_args(config, run_output_dir, evaluation_enabled=evaluation_enabled)
    seed_training_run(config.split_seed)
    model = instantiate_model(config, schema)
    callbacks = []
    if config.early_stopping_enabled and evaluation_enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        )
    trainer = make_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset if evaluation_enabled else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )

    print(f"\n=== Training run: {config.run_name} ===")
    trainer.train()
    if save_model:
        trainer.save_model(str(run_output_dir))
        tokenizer.save_pretrained(str(run_output_dir))

    validation_metrics = {}
    if evaluation_enabled:
        validation_metrics = {key: float(value) for key, value in trainer.evaluate(validation_dataset).items()}
        print("\nValidation metrics:")
        print(json.dumps(validation_metrics, indent=2))
        if write_validation_nervaluate:
            validation_nervaluate_dir = run_output_dir / "nervaluate"
            evaluate_with_nervaluate(
                trainer,
                validation_dataset,
                build_test_records(tokenized_dataset["validation"]),
                schema.id_to_label,
                schema.label_to_id,
                validation_nervaluate_dir,
                suffix="validation",
            )
    elif final_training_mode:
        print("\nFinal training mode: skipped validation during training and will evaluate on test only.")

    run_summary = {
        "run_name": config.run_name,
        "output_dir": str(run_output_dir),
        "training_mode": "final" if final_training_mode else "standard",
        "params": {
            "learning_rate": config.train_learning_rate,
            "per_device_train_batch_size": config.train_batch_size,
            "per_device_eval_batch_size": config.train_batch_size,
            "num_train_epochs": config.train_epochs,
            "weight_decay": config.train_weight_decay,
            "seed": config.split_seed,
            "max_length": config.max_length,
            "model_name": config.model_name,
            "primary_metric": config.primary_metric,
            "excluded_labels": list(config.excluded_labels),
            "early_stopping_enabled": config.early_stopping_enabled,
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_threshold": config.early_stopping_threshold,
        },
        "validation_metrics": validation_metrics,
        "best_metric": float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "completed_epoch": float(trainer.state.epoch) if trainer.state.epoch is not None else None,
    }
    write_json(run_output_dir / "run_summary.json", run_summary)
    persist_label_schema(schema, run_output_dir)

    if save_model:
        print("\nTraining completed. Saved model to", run_output_dir)
    else:
        print("\nTraining completed. Model artifacts were not retained.")
    print("Run hyperparameters:")
    print(json.dumps(run_summary["params"], indent=2))

    if not run_test_evaluation:
        print("\nSkipping test-set evaluation for this run.")
        return run_output_dir

    if save_model:
        test_args = TrainingArguments(
            output_dir=str(run_output_dir / "test_eval"),
            per_device_eval_batch_size=config.train_batch_size,
            dataloader_num_workers=config.dataloader_num_workers,
            report_to="none",
        )
        best_model = AutoModelForTokenClassification.from_pretrained(str(run_output_dir))
        best_trainer = make_trainer(
            model=best_model,
            args=test_args,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=compute_metrics_fn,
        )
    else:
        best_trainer = trainer

    test_metrics = {key: float(value) for key, value in best_trainer.evaluate(test_dataset).items()}
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))
    write_json(run_output_dir / "test_metrics.json", test_metrics)
    prediction_jsonl_path, prediction_csv_path = save_test_predictions(
        best_trainer,
        test_dataset,
        dataset_splits["test"],
        run_output_dir,
        schema,
        metadata_fields=config.optional_string_columns,
    )
    print(
        "\nSaved detailed prediction outputs to:\n"
        f"  JSONL: {prediction_jsonl_path}\n"
        f"  CSV:   {prediction_csv_path}"
    )

    nervaluate_dir = run_output_dir / "nervaluate"
    try:
        nervaluate_results = evaluate_with_nervaluate(
            best_trainer,
            test_dataset,
            build_test_records(tokenized_dataset["test"]),
            schema.id_to_label,
            schema.label_to_id,
            nervaluate_dir,
            suffix="test",
        )
        print("\nNervaluate evaluation written to", nervaluate_dir)
        print(json.dumps(nervaluate_results["trainer_metrics"], indent=2))
    except SystemExit as exc:
        print("\nSkipping Nervaluate evaluation:", exc)

    return run_output_dir
