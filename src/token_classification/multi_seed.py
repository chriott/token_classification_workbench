from __future__ import annotations

import csv
import json
import platform
import statistics
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .config import TrainingConfig
from .utils import ensure_directory, remove_checkpoint_directories, remove_model_weight_files, write_json


AGGREGATE_METRICS = ("precision", "recall", "f1")


def summarize_values(values_by_seed: Mapping[int, float]) -> dict[str, object]:
    ordered = {str(seed): float(value) for seed, value in sorted(values_by_seed.items())}
    values = list(ordered.values())
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values_by_seed": ordered,
    }


def aggregate_nervaluate_results(results_by_seed: Mapping[int, Mapping[str, object]]) -> dict[str, object]:
    if not results_by_seed:
        raise ValueError("At least one nervaluate result is required for aggregation.")

    seeds = sorted(results_by_seed)
    first_result = results_by_seed[seeds[0]]
    labels = sorted(str(label) for label in first_result.get("labels", []))
    scenarios = sorted(str(scenario) for scenario in first_result.get("rollups", {}).keys())
    for seed in seeds[1:]:
        result = results_by_seed[seed]
        if sorted(str(label) for label in result.get("labels", [])) != labels:
            raise ValueError(f"Nervaluate labels differ for seed {seed}.")
        if sorted(str(scenario) for scenario in result.get("rollups", {}).keys()) != scenarios:
            raise ValueError(f"Nervaluate scenarios differ for seed {seed}.")

    scenario_summaries: dict[str, object] = {}
    for scenario in scenarios:
        overall_summary: dict[str, object] = {}
        for averaging in ("micro", "macro"):
            metric_summary: dict[str, object] = {}
            for metric in AGGREGATE_METRICS:
                values = {}
                for seed, result in results_by_seed.items():
                    value = (
                        result.get("rollups", {})
                        .get(scenario, {})
                        .get("overall", {})
                        .get(averaging, {})
                        .get(metric)
                    )
                    if value is not None:
                        values[seed] = float(value)
                if len(values) != len(seeds):
                    raise ValueError(f"Missing overall {scenario} {averaging} {metric} result for one or more seeds.")
                metric_summary[metric] = summarize_values(values)
            overall_summary[averaging] = metric_summary

        label_summaries: dict[str, object] = {}
        for label in labels:
            supports = []
            values_by_metric: dict[str, dict[int, float]] = {metric: {} for metric in AGGREGATE_METRICS}
            for seed, result in results_by_seed.items():
                label_result = result.get("per_tag_results", {}).get(label)
                if not label_result:
                    continue
                supports.append(int(label_result.get("gold_support", 0)))
                if not label_result.get("evaluable", False):
                    continue
                scenario_result = label_result.get("scenarios", {}).get(scenario)
                if scenario_result is None:
                    continue
                for metric in AGGREGATE_METRICS:
                    value = scenario_result.get(metric)
                    if value is not None:
                        values_by_metric[metric][seed] = float(value)

            if len(supports) != len(seeds):
                raise ValueError(f"Missing label result for '{label}' in one or more seeds.")
            gold_support = supports[0]
            if any(support != gold_support for support in supports):
                raise ValueError(f"Gold support for label '{label}' differs across seeds.")
            if gold_support > 0 and any(len(values) != len(seeds) for values in values_by_metric.values()):
                raise ValueError(f"Missing {scenario} metric for label '{label}' in one or more seeds.")
            label_summaries[label] = {
                "gold_support": gold_support,
                "evaluable": gold_support > 0,
                "metrics": {
                    metric: summarize_values(values)
                    for metric, values in values_by_metric.items()
                    if values
                },
            }

        scenario_summaries[scenario] = {
            "overall": overall_summary,
            "labels": label_summaries,
        }

    return {
        "run_count": len(seeds),
        "seeds": seeds,
        "standard_deviation": "sample (n-1)",
        "scenarios": scenario_summaries,
    }


def write_multi_seed_outputs(summary: Mapping[str, object], output_dir: str | Path) -> dict[str, Path]:
    target_dir = ensure_directory(output_dir)
    json_path = write_json(target_dir / "nervaluate_multi_seed.json", summary)
    csv_path = target_dir / "nervaluate_multi_seed.csv"
    text_path = target_dir / "nervaluate_multi_seed.txt"

    rows = []
    for scenario, scenario_summary in summary.get("scenarios", {}).items():
        for averaging, metrics in scenario_summary.get("overall", {}).items():
            for metric, stats in metrics.items():
                rows.append(
                    {
                        "scenario": scenario,
                        "scope": f"overall_{averaging}",
                        "label": "",
                        "gold_support": "",
                        "metric": metric,
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "n": stats["n"],
                        "values_by_seed": json.dumps(stats["values_by_seed"], sort_keys=True),
                    }
                )
        for label, label_summary in scenario_summary.get("labels", {}).items():
            if not label_summary.get("evaluable", False):
                rows.append(
                    {
                        "scenario": scenario,
                        "scope": "label",
                        "label": label,
                        "gold_support": label_summary["gold_support"],
                        "metric": "f1",
                        "mean": "",
                        "std": "",
                        "n": 0,
                        "values_by_seed": "{}",
                    }
                )
                continue
            for metric, stats in label_summary.get("metrics", {}).items():
                rows.append(
                    {
                        "scenario": scenario,
                        "scope": "label",
                        "label": label,
                        "gold_support": label_summary["gold_support"],
                        "metric": metric,
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "n": stats["n"],
                        "values_by_seed": json.dumps(stats["values_by_seed"], sort_keys=True),
                    }
                )

    fieldnames = ["scenario", "scope", "label", "gold_support", "metric", "mean", "std", "n", "values_by_seed"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with text_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Multi-seed nervaluate summary ({summary['run_count']} runs)\n")
        handle.write(f"Seeds: {', '.join(str(seed) for seed in summary['seeds'])}\n")
        handle.write("Standard deviation: sample (n-1)\n")
        for scenario, scenario_summary in summary.get("scenarios", {}).items():
            handle.write(f"\nScenario: {scenario}\n")
            for averaging in ("micro", "macro"):
                f1 = scenario_summary.get("overall", {}).get(averaging, {}).get("f1")
                if f1:
                    handle.write(f"  Overall {averaging} F1: {f1['mean']:.6f} ± {f1['std']:.6f}\n")
            handle.write("  Per-label F1:\n")
            for label, label_summary in scenario_summary.get("labels", {}).items():
                f1 = label_summary.get("metrics", {}).get("f1")
                if f1:
                    handle.write(f"    {label}: {f1['mean']:.6f} ± {f1['std']:.6f}\n")
                else:
                    handle.write(f"    {label}: not evaluable (no gold test examples)\n")

    return {"json": json_path, "csv": csv_path, "text": text_path}


def installed_package_versions() -> dict[str, str]:
    versions = {}
    for package in ("nervaluate", "torch", "transformers"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def run_multi_seed_final_training(
    config: TrainingConfig,
    seeds: Sequence[int],
    *,
    retain_seed: int | None = None,
    pipeline_runner: Callable[..., Path] | None = None,
) -> Path:
    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_seeds:
        raise ValueError("At least one seed must be provided.")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("Seeds must be unique.")
    retained_seed = normalized_seeds[0] if retain_seed is None else int(retain_seed)
    if retained_seed not in normalized_seeds:
        raise ValueError("retain_seed must be one of the requested seeds.")

    if pipeline_runner is None:
        from .training import run_pipeline

        pipeline_runner = run_pipeline

    experiment_dir = ensure_directory(Path(config.output_dir) / config.run_name)
    aggregate_dir = ensure_directory(experiment_dir / "aggregate")
    manifest = {
        "status": "running",
        "seeds": normalized_seeds,
        "retained_model_seed": retained_seed,
        "config": config.to_dict(),
        "python_version": platform.python_version(),
        "package_versions": installed_package_versions(),
        "runs": [],
    }
    manifest_path = aggregate_dir / "run_manifest.json"
    write_json(manifest_path, manifest)

    results_by_seed = {}
    try:
        for seed in normalized_seeds:
            seed_config = config.with_overrides(
                output_dir=str(experiment_dir),
                run_name=f"seed_{seed}",
                split_seed=seed,
            )
            planned_run_dir = Path(seed_config.output_dir) / seed_config.run_name
            try:
                run_output_dir = pipeline_runner(
                    seed_config,
                    final_training_mode=True,
                    save_model=seed == retained_seed,
                )
                result_path = run_output_dir / "nervaluate" / "nervaluate_test.json"
                with result_path.open(encoding="utf-8") as handle:
                    results_by_seed[seed] = json.load(handle)
                manifest["runs"].append(
                    {
                        "seed": seed,
                        "output_dir": str(run_output_dir),
                        "nervaluate_result": str(result_path),
                    }
                )
                write_json(manifest_path, manifest)
            finally:
                remove_checkpoint_directories(planned_run_dir)
                if seed != retained_seed:
                    remove_model_weight_files(planned_run_dir)

        summary = aggregate_nervaluate_results(results_by_seed)
        output_paths = write_multi_seed_outputs(summary, aggregate_dir)
        manifest["status"] = "complete"
        manifest["aggregate_outputs"] = {name: str(path) for name, path in output_paths.items()}
        write_json(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise
    return experiment_dir
