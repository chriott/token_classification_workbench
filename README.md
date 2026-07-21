# Token Classification Workbench

A configurable workflow for training Hugging Face token-classification models from character-span annotations. It includes data splitting, validation, label-coverage checks, evaluation, prediction, and local hyperparameter sweeps.

Labels are discovered from the training data, so the same code can support named-entity recognition and other flat, non-overlapping span-labeling tasks.

## Quick Start

Run these commands from the repository root after cloning it.

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

This installs PyTorch and provides the `token-classification` command.

### 2. Choose a model

Edit [configs/train.example.yaml](configs/train.example.yaml) before preparing long documents. Choose `model_name` and `max_length`; the example uses `FacebookAI/xlm-roberta-base` with a length of 512.

### 3. Create data splits

Place an annotated CSV, JSON, or JSONL file under `data/`, then run:

```bash
token-classification split-data \
  --input-file data/annotated_documents.jsonl \
  --output-dir data/splits \
  --seed 25 \
  --stratify-by constrained_min_labels \
  --chunk-max-length 512 \
  --chunk-tokenizer-model FacebookAI/xlm-roberta-base
```

The chunk tokenizer and maximum length must match the training config. The chunker automatically reserves room for model special tokens. Omit both chunk options only when every input record is already short enough for the selected model.

The example config expects `train.csv`, `validation.csv`, and `test.csv` under `data/splits/`.

### 4. Validate

```bash
token-classification validate-config --config configs/train.example.yaml
token-classification validate-data --config configs/train.example.yaml
token-classification label-coverage --config configs/train.example.yaml
```

### 5. Train

```bash
token-classification train --config configs/train.example.yaml
```

Training artifacts are written under `outputs/training/` by default.

## Modeling Scope

The model schema is derived from the training split. Every label used in validation or test should therefore also occur in training. Span offsets must refer to the exact input text, with an inclusive `start` and exclusive `end`.

## Sweeps and Prediction

Run a local hyperparameter sweep:

```bash
token-classification validate-sweep --config configs/sweeps/sweep.example.yaml
token-classification sweep --config configs/sweeps/sweep.example.yaml
```

Run inference with a trained model:

```bash
token-classification predict \
  --model-path outputs/training/baseline \
  --input-file data/prediction_input.csv
```

## Further Documentation

- [Dataset format and splitting](docs/dataset_format.md)
- [Hyperparameter tuning](docs/hyperparameter_tuning.md)
- [Output artifacts](docs/outputs.md)
- [Development notes](docs/development.md)
