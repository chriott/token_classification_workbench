from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SweepConfig, TrainingConfig
from .data_validation import save_validation_summary, validate_dataset_splits
from .label_coverage import build_label_coverage_report, save_label_coverage_report
from .prediction import run_prediction
from .split_data import split_input_data
from .sweep import run_sweep
from .training import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="token-classification",
        description="Train token-classification models from character-span annotations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run a training job.")
    train_parser.add_argument("--config", required=True, help="Path to a YAML config file.")

    final_train_parser = subparsers.add_parser(
        "final-train",
        help="Train on final training data without validation and evaluate once on the test split.",
    )
    final_train_parser.add_argument("--config", required=True, help="Path to a YAML config file.")

    validate_parser = subparsers.add_parser("validate-config", help="Validate and print a config file.")
    validate_parser.add_argument("--config", required=True, help="Path to a YAML config file.")

    validate_data_parser = subparsers.add_parser("validate-data", help="Validate train/validation/test CSV data for a config.")
    validate_data_parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    validate_data_parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default="all",
        help="Which split to validate. Default: all.",
    )
    validate_data_parser.add_argument(
        "--skip-alignment-check",
        action="store_true",
        help="Skip tokenizer-based span alignment checks.",
    )
    validate_data_parser.add_argument(
        "--output-file",
        help="Optional path for a JSON validation report. Defaults to <output_dir>/data_validation_summary.json.",
    )

    label_coverage_parser = subparsers.add_parser(
        "label-coverage",
        help="Report which labels appear in each split and which are missing.",
    )
    label_coverage_parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    label_coverage_parser.add_argument(
        "--output-file",
        help="Optional path for a JSON coverage report. Defaults to <output_dir>/label_coverage_summary.json.",
    )

    split_data_parser = subparsers.add_parser(
        "split-data",
        help="Split one annotated CSV/JSONL file into train/validation/test files.",
    )
    split_data_parser.add_argument("--input-file", required=True, help="Annotated CSV, JSON, or JSONL input file.")
    split_data_parser.add_argument("--output-dir", required=True, help="Directory for split output files.")
    split_data_parser.add_argument("--text-column", help="Override the input text column name.")
    split_data_parser.add_argument(
        "--spans-column",
        help="Override the source span column name. Defaults to 'spans' with fallback to 'annotations'.",
    )
    split_data_parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio. Default: 0.8.")
    split_data_parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio. Default: 0.1.",
    )
    split_data_parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio. Default: 0.1.")
    split_data_parser.add_argument("--seed", type=int, default=42, help="Shuffle seed. Default: 42.")
    split_data_parser.add_argument(
        "--stratify-by",
        choices=("none", "primary_label", "label_signature", "iterative_multilabel", "constrained_min_labels"),
        default="none",
        help="Optional stratification mode. Default: none.",
    )
    split_data_parser.add_argument(
        "--stratify-column",
        help="Optional column name to stratify by instead of label signatures.",
    )
    split_data_parser.add_argument(
        "--split-manifest-in",
        help="Optional existing split manifest JSON to reuse exact document membership before optional chunking.",
    )
    split_data_parser.add_argument(
        "--min-label-presence",
        type=int,
        default=1,
        help="Minimum document-level presence per label per split for constrained_min_labels. Default: 1.",
    )
    split_data_parser.add_argument(
        "--chunk-max-length",
        type=int,
        help="Optional max token length per output chunk.",
    )
    split_data_parser.add_argument(
        "--chunk-stride",
        type=int,
        default=0,
        help="Token overlap between adjacent chunks. Default: 0.",
    )
    split_data_parser.add_argument(
        "--chunk-tokenizer-model",
        help="Tokenizer model name to use for chunking, required when --chunk-max-length is set.",
    )
    split_data_parser.add_argument(
        "--output-format",
        choices=("csv", "jsonl"),
        default="csv",
        help="Output format for split files. Default: csv.",
    )

    sweep_parser = subparsers.add_parser("sweep", help="Run a hyperparameter sweep.")
    sweep_parser.add_argument("--config", required=True, help="Path to a YAML sweep config file.")

    validate_sweep_parser = subparsers.add_parser("validate-sweep", help="Validate and print a sweep config file.")
    validate_sweep_parser.add_argument("--config", required=True, help="Path to a YAML sweep config file.")

    predict_parser = subparsers.add_parser("predict", help="Run inference with a trained checkpoint on a CSV file.")
    predict_parser.add_argument("--model-path", required=True, help="Path to a trained model directory.")
    predict_parser.add_argument("--input-file", required=True, help="CSV file to run inference on.")
    predict_parser.add_argument("--output-dir", help="Directory for prediction outputs.")
    predict_parser.add_argument("--text-column", help="Override the input text column name.")
    predict_parser.add_argument("--max-length", type=int, help="Override tokenizer max_length.")
    predict_parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size. Default: 8.")
    predict_parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="DataLoader worker count for prediction. Default: 0.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        config = TrainingConfig.from_yaml(args.config)
        config.validate()
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    if args.command == "train":
        config = TrainingConfig.from_yaml(args.config)
        config.validate()
        run_pipeline(config)
        return 0

    if args.command == "final-train":
        config = TrainingConfig.from_yaml(args.config)
        config.validate(require_validation=False)
        run_pipeline(config, final_training_mode=True)
        return 0

    if args.command == "validate-data":
        config = TrainingConfig.from_yaml(args.config)
        config.validate()
        split_names = ("train", "validation", "test") if args.split == "all" else (args.split,)
        summary = validate_dataset_splits(
            config,
            split_names=split_names,
            check_alignment=not args.skip_alignment_check,
        )
        output_file = args.output_file or str(Path(config.output_dir) / "data_validation_summary.json")
        save_validation_summary(summary, output_file)
        print(json.dumps(summary, indent=2))
        print(f"\nSaved validation report to {output_file}")
        return 0

    if args.command == "label-coverage":
        config = TrainingConfig.from_yaml(args.config)
        config.validate()
        report = build_label_coverage_report(config)
        output_file = args.output_file or str(Path(config.output_dir) / "label_coverage_summary.json")
        save_label_coverage_report(report, output_file)
        print(json.dumps(report, indent=2))
        print(f"\nSaved coverage report to {output_file}")
        return 0

    if args.command == "split-data":
        summary = split_input_data(
            input_file=args.input_file,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            text_column=args.text_column,
            spans_column=args.spans_column,
            output_format=args.output_format,
            stratify_by=args.stratify_by,
            stratify_column=args.stratify_column,
            split_manifest_in=args.split_manifest_in,
            min_label_presence=args.min_label_presence,
            chunk_max_length=args.chunk_max_length,
            chunk_stride=args.chunk_stride,
            chunk_tokenizer_model=args.chunk_tokenizer_model,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "validate-sweep":
        sweep_config = SweepConfig.from_yaml(args.config)
        sweep_config.validate()
        print(json.dumps(sweep_config.to_dict(), indent=2))
        return 0

    if args.command == "sweep":
        sweep_config = SweepConfig.from_yaml(args.config)
        sweep_config.validate()
        run_sweep(sweep_config)
        return 0

    if args.command == "predict":
        run_prediction(
            model_path=args.model_path,
            input_file=args.input_file,
            output_dir=args.output_dir,
            text_column=args.text_column,
            max_length=args.max_length,
            batch_size=args.batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
