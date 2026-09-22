# Token Classification Workbench

A configurable workflow for training and evaluating Hugging Face token-classification models from character-span annotations. Use YAML configs to choose a model, prepare data, tune hyperparameters, train final models, and export predictions.

The workbench supports named-entity recognition and other **flat, non-overlapping span-labeling tasks**. Label names are discovered from the training data; no fixed taxonomy or domain-specific metadata is required. Nested and overlapping annotations are outside the current modeling scope.

## Installation

Use Python 3.10 or newer. Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs the package, its dependencies, and development tools, and makes the `token-classification` command available. For usage without development tools, install with `python -m pip install -e .` instead. Training large models requires suitable hardware; batch size, precision, and gradient checkpointing are configurable.

## Prepare your data

Training and splitting accept CSV, JSON, and JSONL files. Start with one complete document per record, a text field, and a list of labeled character spans. For example, one line of a JSONL file could be:

```json
{"document_id":"doc-001","text":"Alice works in Berlin.","spans":[{"start":0,"end":5,"label":"PERSON"},{"start":15,"end":21,"label":"LOCATION"}]}
```

Offsets refer to the exact text: `start` is inclusive and `end` is exclusive. Use an empty `spans` list for documents without annotations. Keep a stable, unique `document_id` for grouping chunks in cross-validation and reconstructing document-level predictions.

Column names can be selected with `text_column` and `spans_column`. Additional metadata can be retained through `optional_string_columns`. See [dataset format and splitting](docs/dataset_format.md) for supported representations and normalization rules.

Datasets, generated splits, model outputs, and caches are local inputs or artifacts and are ignored by Git. Supply your own dataset; the repository includes generic configuration examples.

## Quick start

### 1. Choose a model and training settings

Edit [configs/train.example.yaml](configs/train.example.yaml). It uses `FacebookAI/xlm-roberta-base`, a maximum input length of 512 tokens, and split files under `data/splits/`.

Set the model and input length **before chunking**. The chunk tokenizer must match the model tokenizer, and the chunk length must fit the configured training length and the model's supported input size.

### 2. Split and chunk documents

Save your annotated records to `data/annotated_documents.jsonl`, then run:

```bash
token-classification split-data \
  --input-file data/annotated_documents.jsonl \
  --output-dir data/splits \
  --train-ratio 0.8 \
  --validation-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 25 \
  --stratify-by constrained_min_labels \
  --min-label-presence 1 \
  --chunk-max-length 512 \
  --chunk-stride 128 \
  --chunk-tokenizer-model FacebookAI/xlm-roberta-base \
  --output-format csv
```

This creates `train.csv`, `validation.csv`, and `test.csv`, matching the example config. It also writes a split manifest and label-support statistics. Use `--output-format jsonl` if preferred, and update the config paths accordingly.

Records are split before chunking, so start with complete documents to avoid leaking chunks of the same document across partitions. Chunking reserves space for special tokens and checks the standalone tokenized length. The stride is the overlap between adjacent chunks. Omit chunking only when every record already fits the model input limit.

The constrained splitter aims to include each label in every split where support permits. Labels with too few documents are kept in training; inspect the resulting coverage rather than assuming every label is evaluable on held-out data.

### 3. Validate configuration and annotations

```bash
token-classification validate-config --config configs/train.example.yaml
token-classification validate-data --config configs/train.example.yaml
token-classification label-coverage --config configs/train.example.yaml
```

These commands check configuration, span parsing, offset bounds, overlaps, tokenizer alignment, and label coverage. Inspect the reports before training. Every validation or test label should also occur in training, because training data defines the model's label schema. Config validation alone does not validate dataset contents.

### 4. Run a baseline

```bash
token-classification train --config configs/train.example.yaml
```

Ordinary training uses validation for checkpoint selection and optional early stopping, then evaluates on the configured test set. The example saves its model and reports under `outputs/training/baseline/`. Choose a different `run_name` or output directory for separate baseline runs.

For hyperparameter selection, use the sweep workflow below: its trials skip test evaluation, preserving the test set for final assessment.

## Hyperparameter sweeps

The workbench supports random and grid search. A sweep config contains a training `base_config`, a `search_space`, and an objective metric and direction.

Run the [generic sweep example](configs/sweeps/sweep.example.yaml):

```bash
token-classification validate-sweep --config configs/sweeps/sweep.example.yaml
token-classification sweep --config configs/sweeps/sweep.example.yaml
```

For document-grouped cross-validation, use the [cross-validation example](configs/sweeps/sweep.cross_validation.example.yaml):

```bash
token-classification validate-sweep --config configs/sweeps/sweep.cross_validation.example.yaml
token-classification sweep --config configs/sweeps/sweep.cross_validation.example.yaml
```

The CV example expects `document_id` in the input data. It combines the configured training and validation data into a development pool and assigns complete document groups to folds. All trials reuse those folds; test documents remain outside model selection. Each included label needs support in at least two development document groups. Labels with fewer groups than folds will be absent from some validation folds.

Sweep trials run sequentially. With cross-validation, the number of training fits is the number of trials multiplied by the fold count. Results include an incremental leaderboard, trial summaries, and `best_config.yaml`; CV selection uses the mean objective across folds. Sweep model weights are not retained, so train the selected configuration afterward.

Each sweep creates a separate timestamped output directory. A new invocation starts a new run rather than automatically resuming an interrupted one. If using separate continuation configs, compare their results with the original leaderboard before choosing the overall winner.

See [hyperparameter tuning](docs/hyperparameter_tuning.md) for search-space definitions, CV settings, and available objective metrics.

## Final training and evaluation

After selecting hyperparameters, save the selected training configuration as `configs/final.yaml`, or pass the sweep's generated `best_config.yaml` directly. Check its data paths, output directory, run name, and epoch count before running:

```bash
token-classification final-train --config configs/final.yaml
```

Final training combines the configured training and validation data, trains for the configured epoch count without validation-based early stopping, and evaluates on test data. Each invocation creates a timestamped directory.

To measure variation across training seeds:

```bash
token-classification final-train \
  --config configs/final.yaml \
  --seeds 25 26 27 28 29 \
  --retain-seed 25
```

Each seed is trained and evaluated independently on the same prepared split. Aggregate reports contain means and sample standard deviations for nervaluate metrics. Only the model for the selected retention seed is kept; other seeds retain their reports. Choose the retention seed before examining test results. Without `--retain-seed`, the first requested seed is retained.

## Prediction

Run inference on an input CSV containing text, without requiring gold annotations:

```bash
token-classification predict \
  --model-path outputs/training/baseline \
  --input-file data/prediction_input.csv \
  --output-dir outputs/predictions/baseline
```

The model path above is the baseline from the quick start. For final training, use the actual timestamped model directory, or the retained `seed_<number>` directory for a multi-seed run.

Prediction reads the saved tokenizer and training settings. Prepare long inputs as model-compatible chunks before inference; the prediction command truncates over-length records rather than automatically creating overlapping chunks. When document IDs and chunk offsets are available, additional document-level exports restore original offsets and remove exact duplicate spans from overlapping chunks.

## Configuration and outputs

Useful configuration controls include:

| Purpose | Settings |
|---|---|
| Model and inputs | `model_name`, `max_length`, train/validation/test file paths |
| Annotation columns | `text_column`, `spans_column`, `optional_string_columns` |
| Optimization | `train_learning_rate`, `train_batch_size`, `train_epochs`, `train_weight_decay`, `warmup_ratio` |
| Memory and precision | `gradient_accumulation_steps`, `gradient_checkpointing`, `fp16`, `bf16` |
| Selection | `primary_metric`, early-stopping settings, sweep objective |
| Label filtering | `excluded_labels` in training configs; `cross_validation.excluded_labels` for CV-only filtering |

Enable mixed precision only on compatible hardware, and choose either FP16 or BF16. Label exclusions remove matching spans while keeping the documents; document any exclusions when interpreting metrics.

To compare models or input lengths, reuse the same split manifest with `split-data --split-manifest-in`, prepare separate tokenizer-compatible chunks, and point each config to its own files. Matching token limits does not produce identical text boundaries across different tokenizers. Keep document membership and CV settings consistent for comparisons.

| Workflow | Main artifacts |
|---|---|
| Data preparation and validation | Split manifest, label statistics, validation and coverage reports |
| Training | Saved model/tokenizer, label mappings, `config_used.yaml`, `run_summary.json`, test metrics and prediction exports |
| Sweeps | `sweep_config_used.yaml`, `summary.json`, `leaderboard.csv`, `best_config.yaml`; CV fold manifests and trial summaries |
| Multi-seed final training | Per-seed reports and aggregate nervaluate JSON, CSV, and text reports |
| Prediction | Row-level and document-level JSONL/CSV exports plus `prediction_summary.json` |

Evaluation includes token-classification metrics and nervaluate strict, entity-type, and partial-match results, with micro/macro and per-label reporting. Labels without gold test examples are marked as not evaluable in the nervaluate reports.

## Development and documentation

Run the local checks after installing the development dependencies:

```bash
pytest
ruff check .
```

Use `token-classification --help` or `token-classification <command> --help` for command options.

- [Dataset format and splitting](docs/dataset_format.md)
- [Hyperparameter tuning](docs/hyperparameter_tuning.md)
- [Output artifacts](docs/outputs.md)
- [Development notes](docs/development.md)
