from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from .config import SweepConfig, SweepParameter, dump_config_yaml, dump_yaml_mapping
from .cross_validation import aggregate_fold_nervaluate_results, prepare_cross_validation_folds
from .training import run_pipeline
from .utils import ensure_directory, remove_checkpoint_directories, remove_model_weight_files, write_json


def cast_sampled_value(parameter: SweepParameter, value: Any) -> Any:
    if parameter.type == "int":
        return int(round(value))
    if parameter.type == "float":
        return float(value)
    if parameter.type == "bool":
        return bool(value)
    if parameter.type == "str":
        return str(value)
    return value


def sample_parameter_value(parameter: SweepParameter, rng: random.Random) -> Any:
    if parameter.values is not None:
        return cast_sampled_value(parameter, rng.choice(parameter.values))

    if parameter.min is None or parameter.max is None:
        raise ValueError("Range-based sweep parameters must define both min and max.")

    if parameter.type == "int":
        if parameter.step is not None:
            values = list(range(int(parameter.min), int(parameter.max) + 1, int(parameter.step)))
            return rng.choice(values)
        return rng.randint(int(parameter.min), int(parameter.max))

    if parameter.log:
        sampled = math.exp(rng.uniform(math.log(float(parameter.min)), math.log(float(parameter.max))))
    else:
        sampled = rng.uniform(float(parameter.min), float(parameter.max))

    if parameter.step is not None:
        base = float(parameter.min)
        step = float(parameter.step)
        sampled = base + round((sampled - base) / step) * step
        sampled = min(float(parameter.max), max(float(parameter.min), sampled))

    return cast_sampled_value(parameter, sampled)


def sample_trial_overrides(sweep_config: SweepConfig, rng: random.Random) -> dict[str, Any]:
    return {
        name: sample_parameter_value(parameter, rng)
        for name, parameter in sweep_config.search_space.items()
    }


def enumerate_grid_trial_overrides(sweep_config: SweepConfig) -> list[dict[str, Any]]:
    parameter_names = list(sweep_config.search_space.keys())
    value_lists: list[list[Any]] = []
    for name in parameter_names:
        parameter = sweep_config.search_space[name]
        if parameter.values is None:
            raise ValueError(f"Grid search requires explicit values for parameter '{name}'.")
        value_lists.append([cast_sampled_value(parameter, value) for value in parameter.values])
    return [
        dict(zip(parameter_names, combination))
        for combination in itertools.product(*value_lists)
    ]


def score_for_sorting(metric_value: float | None, objective_mode: str) -> float:
    if metric_value is None:
        return float("-inf") if objective_mode == "max" else float("inf")
    return metric_value


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


remove_trial_checkpoints = remove_checkpoint_directories


def write_leaderboard_csv(path: str | Path, trial_rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "trial_number",
        "run_name",
        "status",
        "objective_metric",
        "objective_value",
        "objective_std",
        "objective_min",
        "objective_max",
        "completed_folds",
        "run_output_dir",
        "overrides_json",
        "error",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_rows)
    return target


def persist_sweep_state(
    sweep_config: SweepConfig,
    sweep_root: str | Path,
    objective_metric: str,
    planned_num_trials: int,
    trial_results: list[dict[str, Any]],
) -> dict[str, Any]:
    sweep_root = Path(sweep_root)
    completed_trials = [row for row in trial_results if row["status"] == "completed"]
    reverse = sweep_config.objective_mode == "max"
    completed_trials.sort(
        key=lambda row: score_for_sorting(row["objective_value"], sweep_config.objective_mode),
        reverse=reverse,
    )

    best_trial = completed_trials[0] if completed_trials else None
    leaderboard_rows = []
    ranked_run_names = {row["run_name"]: index + 1 for index, row in enumerate(completed_trials)}
    for row in trial_results:
        leaderboard_rows.append(
            {
                "rank": ranked_run_names.get(row["run_name"], ""),
                "trial_number": row["trial_number"],
                "run_name": row["run_name"],
                "status": row["status"],
                "objective_metric": row["objective_metric"],
                "objective_value": row["objective_value"],
                "objective_std": row.get("objective_std", ""),
                "objective_min": row.get("objective_min", ""),
                "objective_max": row.get("objective_max", ""),
                "completed_folds": row.get("completed_folds", ""),
                "run_output_dir": row["run_output_dir"],
                "overrides_json": json.dumps(row["overrides"], sort_keys=True),
                "error": row["error"] or "",
            }
        )

    summary = {
        "name": sweep_config.name,
        "output_dir": str(sweep_root),
        "num_trials": planned_num_trials,
        "requested_num_trials": sweep_config.num_trials,
        "completed_trials": len(completed_trials),
        "failed_trials": len(trial_results) - len(completed_trials),
        "search_strategy": sweep_config.search_strategy,
        "objective_metric": objective_metric,
        "objective_mode": sweep_config.objective_mode,
        "best_trial": best_trial,
        "trials": trial_results,
    }
    write_json(sweep_root / "summary.json", summary)
    write_leaderboard_csv(sweep_root / "leaderboard.csv", leaderboard_rows)

    if best_trial is not None:
        best_config = sweep_config.base_config.with_overrides(
            **best_trial["overrides"],
            output_dir=str(sweep_root),
            run_name=best_trial["run_name"],
        )
        dump_config_yaml(sweep_root / "best_config.yaml", best_config)

    return summary


def run_sweep(sweep_config: SweepConfig) -> Path:
    sweep_config.validate()
    rng = random.Random(sweep_config.seed)
    objective_metric = sweep_config.objective_metric or sweep_config.base_config.primary_metric

    sweep_root = ensure_directory(Path(sweep_config.output_dir) / sweep_config.name)
    dump_yaml_mapping(sweep_root / "sweep_config_used.yaml", sweep_config.to_dict())
    fold_specs = None
    if sweep_config.cross_validation is not None:
        fold_specs = prepare_cross_validation_folds(
            sweep_config.base_config,
            sweep_config.cross_validation,
            sweep_root / "folds",
        )

    if sweep_config.search_strategy == "grid":
        planned_overrides = enumerate_grid_trial_overrides(sweep_config)
        planned_overrides = planned_overrides[: sweep_config.num_trials]
    else:
        planned_overrides = [
            sample_trial_overrides(sweep_config, rng)
            for _ in range(sweep_config.num_trials)
        ]

    trial_results: list[dict[str, Any]] = []
    for trial_number, overrides in enumerate(planned_overrides, start=1):
        run_name = f"trial_{trial_number:03d}"
        trial_output_dir = sweep_root / run_name
        trial_config = sweep_config.base_config.with_overrides(
            **overrides,
            output_dir=str(sweep_root),
            run_name=run_name,
        )

        print(f"\n=== Sweep trial {trial_number}/{len(planned_overrides)}: {run_name} ===")
        print(json.dumps(overrides, indent=2, sort_keys=True))

        trial_config_path = dump_config_yaml(sweep_root / f"{run_name}_config.yaml", trial_config)
        try:
            if fold_specs is None:
                run_output_dir = run_pipeline(
                    trial_config,
                    run_test_evaluation=False,
                    save_model=False,
                )
                run_summary = load_json(run_output_dir / "run_summary.json")
                validation_metrics = run_summary.get("validation_metrics", {})
                objective_value = validation_metrics.get(objective_metric)
                objective_std = None
                objective_min = objective_value
                objective_max = objective_value
                completed_folds = None
            else:
                fold_rows = []
                nervaluate_by_fold = {}
                for fold in fold_specs:
                    fold_run_name = f"fold_{fold.index}"
                    fold_output_dir = trial_output_dir / fold_run_name
                    fold_config = trial_config.with_overrides(
                        output_dir=str(trial_output_dir),
                        run_name=fold_run_name,
                        train_file=fold.train_files,
                        validation_file=fold.validation_file,
                        split_seed=sweep_config.cross_validation.seed + fold.index,
                    )
                    try:
                        run_output_dir = run_pipeline(
                            fold_config,
                            run_test_evaluation=False,
                            save_model=False,
                            write_validation_nervaluate=True,
                        )
                        fold_summary = load_json(run_output_dir / "run_summary.json")
                        fold_metrics = fold_summary.get("validation_metrics", {})
                        fold_objective = fold_metrics.get(objective_metric)
                        if fold_objective is None:
                            raise ValueError(
                                f"Objective metric '{objective_metric}' was not produced for fold {fold.index}."
                            )
                        fold_rows.append(
                            {
                                "fold": fold.index,
                                "objective_value": float(fold_objective),
                                "validation_metrics": fold_metrics,
                                "completed_epoch": fold_summary.get("completed_epoch"),
                                "best_metric": fold_summary.get("best_metric"),
                            }
                        )
                        nervaluate_by_fold[fold.index] = load_json(
                            run_output_dir / "nervaluate" / "nervaluate_validation.json"
                        )
                    finally:
                        remove_trial_checkpoints(fold_output_dir)
                        remove_model_weight_files(fold_output_dir)

                objective_values = [row["objective_value"] for row in fold_rows]
                objective_value = statistics.mean(objective_values)
                objective_std = statistics.stdev(objective_values) if len(objective_values) > 1 else 0.0
                objective_min = min(objective_values)
                objective_max = max(objective_values)
                completed_folds = len(fold_rows)
                validation_metric_names = set.intersection(
                    *(set(row["validation_metrics"]) for row in fold_rows)
                )
                validation_metrics = {
                    metric: statistics.mean(float(row["validation_metrics"][metric]) for row in fold_rows)
                    for metric in sorted(validation_metric_names)
                }
                cross_validation_summary = {
                    "objective_metric": objective_metric,
                    "objective_mean": objective_value,
                    "objective_std": objective_std,
                    "objective_min": objective_min,
                    "objective_max": objective_max,
                    "folds": fold_rows,
                    "nervaluate": aggregate_fold_nervaluate_results(nervaluate_by_fold),
                }
                write_json(trial_output_dir / "trial_summary.json", cross_validation_summary)
                run_output_dir = trial_output_dir
            trial_results.append(
                {
                    "trial_number": trial_number,
                    "run_name": run_name,
                    "status": "completed",
                    "objective_metric": objective_metric,
                    "objective_value": objective_value,
                    "objective_std": objective_std,
                    "objective_min": objective_min,
                    "objective_max": objective_max,
                    "completed_folds": completed_folds,
                    "run_output_dir": str(run_output_dir),
                    "config_path": str(trial_config_path),
                    "overrides": overrides,
                    "validation_metrics": validation_metrics,
                    "error": None,
                }
            )
        except Exception as exc:
            trial_results.append(
                {
                    "trial_number": trial_number,
                    "run_name": run_name,
                    "status": "failed",
                    "objective_metric": objective_metric,
                    "objective_value": None,
                    "objective_std": None,
                    "objective_min": None,
                    "objective_max": None,
                    "completed_folds": 0 if fold_specs is not None else None,
                    "run_output_dir": str(sweep_root / run_name),
                    "config_path": str(trial_config_path),
                    "overrides": overrides,
                    "validation_metrics": {},
                    "error": str(exc),
                }
            )
            print(f"Trial {run_name} failed: {exc}")
        finally:
            remove_trial_checkpoints(trial_output_dir)
            remove_model_weight_files(trial_output_dir)
        persist_sweep_state(
            sweep_config=sweep_config,
            sweep_root=sweep_root,
            objective_metric=objective_metric,
            planned_num_trials=len(planned_overrides),
            trial_results=trial_results,
        )

    summary = persist_sweep_state(
        sweep_config=sweep_config,
        sweep_root=sweep_root,
        objective_metric=objective_metric,
        planned_num_trials=len(planned_overrides),
        trial_results=trial_results,
    )
    best_trial = summary["best_trial"]
    if best_trial is not None:
        print(
            "\nBest trial:\n"
            f"  run_name: {best_trial['run_name']}\n"
            f"  {objective_metric}: {best_trial['objective_value']}"
        )
    else:
        print("\nNo successful trials completed.")

    return sweep_root
