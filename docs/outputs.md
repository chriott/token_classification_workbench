# Output Artifacts

Each training run writes to:

```text
<output_dir>/<run_name>/
```

## Core Files

`config_used.yaml`

- The exact config used for the run.

`data_validation_summary.json`

- Data quality summary written by `validate-data`.
- Includes parse errors, out-of-bounds spans, overlap counts, and optional tokenizer alignment failures.

`run_summary.json`

- High-level run metadata, chosen hyperparameters, and validation metrics.
- Includes the early stopping settings used for the run.

`test_metrics.json`

- Aggregate test-set metrics from the Hugging Face trainer evaluation pass.

`label2id.json`, `id2label.json`, `bio_labels.json`

- Label metadata saved alongside the trained model.

## Prediction Exports

`test_predictions_detailed.jsonl`

- One JSON object per test example.
- Includes full text, predicted spans, gold spans, and TP/FP/FN groupings.

`test_span_classification.csv`

- Flat row-oriented export of TP/FP/FN spans with metadata columns.

## Inference Outputs

`outputs/predictions/<model>_<input>/`

- `prediction_summary.json`
- `predictions.jsonl`
- `predicted_spans.csv`

These come from the `predict` CLI command and do not require gold labels in the input CSV.

## Nervaluate Outputs

`nervaluate/`

- `nervaluate_test.json`
- `nervaluate_test.txt`

These contain strict, entity-type, and partial-match summaries, overall micro/macro rollups, and per-label results for every label discovered in the training data.

The JSON metrics are read directly from nervaluate's structured evaluation results; the text report is retained for
human inspection. Labels without gold examples in the test split are marked as not evaluable and are excluded from
macro averaging.

## Multi-Seed Final Training

Running `final-train` with `--seeds` creates one complete run directory per seed and an aggregate directory:

```text
<output_dir>/<run_name>/
├── seed_42/
├── seed_43/
├── seed_44/
├── seed_45/
├── seed_46/
└── aggregate/
    ├── run_manifest.json
    ├── nervaluate_multi_seed.json
    ├── nervaluate_multi_seed.csv
    └── nervaluate_multi_seed.txt
```

Aggregation is kept separate for strict, entity-type, and partial matching. It reports the mean and sample standard
deviation (`n - 1`) of precision, recall, and F1 for overall micro, overall macro, and every evaluable label. Raw values
are retained by seed. Labels with no gold test examples are listed as not evaluable rather than assigned an F1 of zero.
