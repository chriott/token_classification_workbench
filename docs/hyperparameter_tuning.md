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
- per-trial configs and model-output directories

Trials are ranked only on validation metrics and intentionally skip test evaluation. Use the test split once, after selecting the final configuration.
