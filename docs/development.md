# Development Notes

The supported entrypoint is `python ipi.py <command>` from the repository root. Routine training should be configured through YAML rather than by editing Python files.

## Package Layout

- `config.py`: training and sweep configuration models
- `data.py`: data loading, column detection, and span normalization
- `data_validation.py`: dataset QA and tokenizer-alignment checks
- `label_coverage.py`: label counts and cross-split coverage reports
- `labels.py`: dynamic label-schema creation and BIO conversion
- `split_data.py`: splitting, stratification, manifests, and token chunking
- `training.py`: model and trainer setup plus the end-to-end pipeline
- `evaluation.py`: metrics and detailed evaluation exports
- `prediction.py`: inference with a trained checkpoint
- `sweep.py`: local random and grid searches
- `cli.py`: command-line interface

## Design Choices

- Label schemas are derived from the training data, allowing collaborators to use different label names without code changes.
- Validation stays separate during normal training and drives early stopping and sweep selection.
- Sweep trials skip test evaluation so the test set remains a final holdout.
- Data, generated splits, outputs, and checkpoints remain local and are excluded from Git.
- Tests create their own temporary data and do not depend on repository datasets.

The original `bert_ipi_detection_roberta_train.py` remains as a legacy reference. New work should target the modules under `src/ipi_training/`.

## Local Checks

```bash
pytest
ruff check .
python ipi.py validate-config --config configs/train.example.yaml
python ipi.py validate-sweep --config configs/sweeps/sweep.example.yaml
```

Dataset-dependent commands require each collaborator’s local files under `data/` or equivalent paths configured in YAML.
