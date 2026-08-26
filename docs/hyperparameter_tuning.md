# Hyperparameter Tuning

The repository supports local random and grid searches through the `sweep` command. Start with `configs/sweeps/sweep.example.yaml` and update its data paths, model, hardware settings, and search space.

Validate the sweep definition:

```bash
token-classification validate-sweep --config configs/sweeps/sweep.example.yaml
```

Run it:

```bash
token-classification sweep --config configs/sweeps/sweep.example.yaml
```

## Sweep Structure

The top-level fields are:

- `name`: sweep name and output subdirectory
- `output_dir`: parent directory for sweep artifacts
- `num_trials`: maximum number of trials
- `search_strategy`: `random` or `grid`
- `seed`: sampling seed
- `objective_metric`: validation metric used to rank trials
- `objective_mode`: `max` or `min`
- `base_config`: a complete normal training configuration
- `search_space`: training fields to vary

Random searches can sample discrete values or numeric ranges:

```yaml
search_space:
  train_learning_rate:
    min: 1.0e-5
    max: 8.0e-5
    type: float
    log: true
  train_batch_size:
    values: [4, 8, 16]
    type: int
```

Grid searches require an explicit `values` list for every parameter. The Cartesian product is limited by `num_trials`.

## Outputs

Each sweep writes to `outputs/sweeps/<name>/`. Important artifacts include:

- `sweep_config_used.yaml`
- `summary.json`
- `leaderboard.csv`
- `best_config.yaml`
- per-trial configs and metric-output directories

Trials are ranked only on validation metrics and intentionally skip test evaluation. Use the test split once, after selecting the final configuration.
Checkpoints are used temporarily for early stopping and restoring each trial's best epoch, then deleted after the trial finishes.
Sweep trials do not retain model weights; run `final-train` with `best_config.yaml` to train and save the selected model.

## Grouped Cross-Validation Sweeps

Use `configs/sweeps/sweep.cross_validation.example.yaml` to evaluate every sampled hyperparameter combination across
fixed folds. Cross-validation combines the configured training and validation files; the test file remains untouched.

```yaml
objective_metric: eval_nervaluate_partial_micro_f1

cross_validation:
  folds: 5
  seed: 211
  group_column: document_id
  stratify_by: iterative_multilabel
```

`group_column` is required. All rows or chunks with the same value are assigned to the same fold, preventing content
from one source document from appearing in both fold training and validation data. Fold assignments are generated once,
saved under `folds/fold_manifest.json`, and reused by every trial.

Each trial writes `trial_summary.json` containing the fold-level objective mean, sample standard deviation, minimum,
maximum, individual fold values, and pooled out-of-fold nervaluate results. The leaderboard ranks trials by the mean
fold objective. A label occurring in fewer than five document groups is reported as sparse across folds; a label in
only one group is rejected because it would disappear from the training portion of one fold.

Fold runs are sequential. Model checkpoints are temporary, bounded by `save_total_limit`, and deleted immediately after
each fold. No sweep or fold model weights are retained.

Nervaluate validation metrics are available as checkpoint and sweep objectives, including:

- `eval_nervaluate_partial_micro_f1`
- `eval_nervaluate_partial_macro_f1`
- `eval_nervaluate_ent_type_micro_f1`
- `eval_nervaluate_ent_type_macro_f1`
- `eval_nervaluate_strict_micro_f1`
- `eval_nervaluate_strict_macro_f1`
