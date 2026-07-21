# Token Classification Workbench

A configurable workflow for training token-classification models from character-span annotations. It supports data splitting, validation, label-coverage checks, Hugging Face model training, evaluation, prediction, and local hyperparameter sweeps.

Label names are discovered from the training data, so the same pipeline can be used for named-entity recognition, domain-specific entity extraction, and other non-overlapping span-labeling tasks without changing Python code.

Datasets, generated splits, checkpoints, and outputs stay local and are excluded from Git.

## Scope

This project expects text with labeled character spans and converts those spans to BIO token labels. It is designed for flat, non-overlapping token classification. Nested or overlapping entities, document classification, relation extraction, and text generation require different modeling approaches.

## Requirements

- Python 3.10 or newer
- A PyTorch installation suitable for your CPU, CUDA, or Apple Silicon setup
- A fresh virtual environment

Create an environment, install the appropriate PyTorch build for your machine, and install the workbench:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch for your hardware: https://pytorch.org/get-started/locally/
pip install -e ".[dev]"
```

PyTorch is intentionally not declared as a project dependency because its correct build depends on the available hardware. The editable install provides the `token-classification` command. `python token_classifier.py` remains available as a repository-local launcher.

Verify the installation:

```bash
token-classification --help
token-classification validate-config --config configs/train.example.yaml
```

## Repository Layout

```text
.
├── configs/
│   ├── train.example.yaml
│   └── sweeps/sweep.example.yaml
├── data/                 # local and ignored, except for its README
├── docs/
├── examples/
├── src/token_classification/
├── tests/
├── pyproject.toml
└── token_classifier.py
```

## Prepare Data

The pipeline accepts annotated CSV, JSON, or JSONL records. Each record needs a text field and character-offset spans containing `start`, `end`, and `label`.

Split an annotated source file:

```bash
token-classification split-data \
  --input-file data/annotated_documents.jsonl \
  --output-dir data/splits \
  --seed 42 \
  --stratify-by constrained_min_labels \
  --min-label-presence 1
```

The example configurations expect:

```text
data/splits/train.csv
data/splits/validation.csv
data/splits/test.csv
```

Change those paths when using a different layout. See [Dataset Format](docs/dataset_format.md) for the schema and splitting options.

## Validate Labels and Spans

The model schema is derived from labels in the training split. Every label evaluated in validation or test should therefore also appear in training.

```bash
token-classification validate-data --config configs/train.example.yaml
token-classification label-coverage --config configs/train.example.yaml
```

## Train

Copy or edit `configs/train.example.yaml` to choose the model, data paths, output directory, batch size, metadata columns, and other hyperparameters.

```bash
token-classification validate-config --config configs/train.example.yaml
token-classification train --config configs/train.example.yaml
```

The example uses `fp16: false` for portability. Enable it only when supported by your hardware and PyTorch setup.

## Tune Hyperparameters

Edit the base configuration and search space in `configs/sweeps/sweep.example.yaml`, then run:

```bash
token-classification validate-sweep --config configs/sweeps/sweep.example.yaml
token-classification sweep --config configs/sweeps/sweep.example.yaml
```

Sweep trials are ranked using validation metrics and do not evaluate the test set. See [Hyperparameter Tuning](docs/hyperparameter_tuning.md).

## Predict

After training, run inference on a CSV file containing text:

```bash
token-classification predict \
  --model-path outputs/training/baseline \
  --input-file data/prediction_input.csv
```

See [Output Artifacts](docs/outputs.md) for generated files and [Development Notes](docs/development.md) for the package layout.

## Data Handling

The `data/` directory is ignored by Git apart from `data/README.md`. Do not force-add sensitive, confidential, or restrictively licensed data. Share data only through an approved storage and transfer process.
