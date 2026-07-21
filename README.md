# IPI/PHI Model Training

This repository provides a configurable token-classification workflow for training IPI/PHI detection models. It includes data splitting, validation, label-coverage checks, training, evaluation, prediction, and local hyperparameter sweeps.

Datasets, generated splits, model checkpoints, and outputs are intentionally excluded from Git. Each collaborator supplies their own annotated data and label set.

## Requirements

- Python 3.10 or 3.11
- A PyTorch installation suitable for your CPU, CUDA, or Apple Silicon setup
- A fresh virtual environment

Create the environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate torch build for your machine first.
pip install -r requirements.txt
```

PyTorch is not pinned in `requirements.txt` because its correct installation depends on the available hardware.

## Repository Layout

```text
.
├── configs/
│   ├── train.example.yaml
│   └── sweeps/sweep.example.yaml
├── data/                 # local and ignored, except for its README
├── docs/
├── examples/
├── src/ipi_training/
├── tests/
├── ipi.py
└── requirements.txt
```

## Prepare Your Data

The pipeline accepts annotated CSV, JSON, or JSONL records. Each record needs a text field and character-offset spans containing `start`, `end`, and `label`.

To split an annotated source file:

```bash
python ipi.py split-data \
  --input-file data/annotated_documents.jsonl \
  --output-dir data/splits \
  --seed 42 \
  --stratify-by constrained_min_labels \
  --min-label-presence 1
```

The example training and sweep configs expect:

```text
data/splits/train.csv
data/splits/validation.csv
data/splits/test.csv
```

Change those paths in the YAML files if you use a different layout. See [Dataset Format](docs/dataset_format.md) for the complete schema and splitting guidance.

## Use Your Own Labels

Label names are not hard-coded. The training pipeline discovers them from the spans in the training split and creates the corresponding BIO labels automatically.

Every label evaluated in validation or test should also occur in training. Check this before training:

```bash
python ipi.py label-coverage --config configs/train.example.yaml
```

## Configure and Train

Copy or edit `configs/train.example.yaml` to choose the model, data paths, output directory, batch size, and other hyperparameters.

Validate the YAML without loading data or downloading a model:

```bash
python ipi.py validate-config --config configs/train.example.yaml
```

Validate the configured datasets:

```bash
python ipi.py validate-data --config configs/train.example.yaml
```

Run training:

```bash
python ipi.py train --config configs/train.example.yaml
```

The example has `fp16: false` for portability. Enable it only when supported by your hardware and PyTorch setup.

## Run a Sweep

Edit the base config and search space in `configs/sweeps/sweep.example.yaml`, then run:

```bash
python ipi.py validate-sweep --config configs/sweeps/sweep.example.yaml
python ipi.py sweep --config configs/sweeps/sweep.example.yaml
```

Sweep trials are ranked using validation metrics and do not evaluate the test set. See [Hyperparameter Tuning](docs/hyperparameter_tuning.md).

## Run Prediction

After training, run inference on a CSV file containing text:

```bash
python ipi.py predict \
  --model-path outputs/training/baseline \
  --input-file data/prediction_input.csv
```

See [Output Artifacts](docs/outputs.md) for generated files and [Development Notes](docs/development.md) for the code layout.

## Data Privacy

The `data/` directory is ignored by Git, apart from `data/README.md`. Do not force-add patient, confidential, or licensed datasets. Share data only through an approved data-transfer and storage process.
