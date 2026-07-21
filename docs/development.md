# Development Notes

The supported entrypoint is `token-classification <command>` after an editable install. The repository-local `token-classification <command>` launcher provides the same interface. Routine training should be configured through YAML rather than by editing Python files.

## Package Layout

- `config.py`: training and sweep configuration models
- `data.py`: data loading, column detection, and span normalization
- `data_validation.py`: dataset QA and tokenizer-alignment checks
- `label_coverage.py`: label counts and cross-split coverage reports
- `labels.py`: dynamic label-schema creation and BIO conversion
- `split_data.py`: splitting, stratification, manifests, and token chunking
- `training.py`: model and trainer setup plus the end-to-end pipeline
- `evaluation.py`: overall and per-label metrics plus detailed evaluation exports
- `prediction.py`: inference with a trained checkpoint
- `sweep.py`: local random and grid searches
- `cli.py`: command-line interface

## Design Choices

- Label schemas are derived from the training data, allowing collaborators to use different label names without code changes.
- Optional metadata exports are controlled by `optional_string_columns`; no domain-specific metadata fields are assumed.
- Validation stays separate during normal training and drives early stopping and sweep selection.
- Sweep trials skip test evaluation so the test set remains a final holdout.
- Data, generated splits, outputs, and checkpoints remain local and are excluded from Git.
- Tests create their own temporary data and do not depend on repository datasets.

## Local Checks

```bash
pytest
ruff check .
token-classification validate-config --config configs/train.example.yaml
token-classification validate-sweep --config configs/sweeps/sweep.example.yaml
```

Dataset-dependent commands require each collaborator’s local files under `data/` or equivalent paths configured in YAML.
