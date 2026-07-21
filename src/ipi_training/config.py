from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass
class TrainingConfig:
    model_name: str = "roberta-large"
    output_dir: str = "outputs/roberta_large"
    run_name: str = "baseline"
    max_length: int = 512
    split_seed: int = 42
    train_learning_rate: float = 5e-5
    train_batch_size: int = 16
    train_epochs: float = 30
    train_weight_decay: float = 0.01
    text_column: str | None = None
    spans_column: str = "spans"
    optional_string_columns: tuple[str, ...] = ("subject_id", "hadm_id", "chartdate")
    train_file: str = "data/train.csv"
    validation_file: str | None = "data/validation.csv"
    test_file: str = "data/test.csv"
    logging_steps: int = 200
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    save_total_limit: int = 1
    warmup_ratio: float = 0.06
    dataloader_num_workers: int = 4
    primary_metric: str = "eval_f1_macro"
    report_target: str = "none"
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    def validate(self, *, require_validation: bool = True) -> None:
        if not self.train_file:
            raise ValueError("train_file must be set.")
        if require_validation and not self.validation_file:
            raise ValueError("validation_file must be set.")
        if not self.test_file:
            raise ValueError("test_file must be set.")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be greater than zero.")
        if self.max_length <= 0:
            raise ValueError("max_length must be greater than zero.")
        if self.train_epochs <= 0:
            raise ValueError("train_epochs must be greater than zero.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1.")
        if self.early_stopping_threshold < 0:
            raise ValueError("early_stopping_threshold must be zero or greater.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> "TrainingConfig":
        payload = dict(overrides)
        optional_columns = payload.get("optional_string_columns")
        if optional_columns is not None:
            payload["optional_string_columns"] = tuple(optional_columns)
        return replace(self, **payload)

    @classmethod
    def from_mapping(cls, raw_config: dict[str, Any]) -> "TrainingConfig":
        payload = dict(raw_config)
        optional_columns = payload.get("optional_string_columns")
        if optional_columns is not None:
            payload["optional_string_columns"] = tuple(optional_columns)
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        raw_config = load_yaml_mapping(path)
        return cls.from_mapping(raw_config)


@dataclass
class SweepParameter:
    values: list[Any] | None = None
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    type: str | None = None
    log: bool = False

    def validate(self, name: str) -> None:
        has_values = self.values is not None
        has_range = self.min is not None and self.max is not None
        if has_values == has_range:
            raise ValueError(f"Sweep parameter '{name}' must define exactly one of 'values' or 'min'/'max'.")
        if self.type not in (None, "int", "float", "str", "bool"):
            raise ValueError(f"Sweep parameter '{name}' has unsupported type: {self.type}")
        if has_range and self.min > self.max:
            raise ValueError(f"Sweep parameter '{name}' has min greater than max.")
        if self.log and has_range and (self.min <= 0 or self.max <= 0):
            raise ValueError(f"Sweep parameter '{name}' must have positive bounds when log sampling is enabled.")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Sweep parameter '{name}' must have a positive step.")


@dataclass
class SweepConfig:
    name: str
    base_config: TrainingConfig
    search_space: dict[str, SweepParameter]
    output_dir: str = "outputs/sweeps"
    num_trials: int = 10
    search_strategy: str = "random"
    seed: int = 42
    objective_metric: str | None = None
    objective_mode: str = "max"

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Sweep name must be set.")
        if self.num_trials < 1:
            raise ValueError("num_trials must be at least 1.")
        if self.search_strategy not in ("random", "grid"):
            raise ValueError("search_strategy must be either 'random' or 'grid'.")
        if self.objective_mode not in ("max", "min"):
            raise ValueError("objective_mode must be either 'max' or 'min'.")
        if not self.search_space:
            raise ValueError("search_space must contain at least one parameter.")
        self.base_config.validate()
        valid_training_fields = {field.name for field in fields(TrainingConfig)}
        for name, parameter in self.search_space.items():
            if name not in valid_training_fields:
                raise ValueError(f"Sweep parameter '{name}' is not a field on TrainingConfig.")
            parameter.validate(name)
            if self.search_strategy == "grid" and parameter.values is None:
                raise ValueError(
                    f"Sweep parameter '{name}' must define explicit 'values' for grid search."
                )
        objective_metric = self.objective_metric or self.base_config.primary_metric
        if not objective_metric:
            raise ValueError("objective_metric must be set either on the sweep or base config.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_dir": self.output_dir,
            "num_trials": self.num_trials,
            "search_strategy": self.search_strategy,
            "seed": self.seed,
            "objective_metric": self.objective_metric,
            "objective_mode": self.objective_mode,
            "base_config": self.base_config.to_dict(),
            "search_space": {
                name: {
                    key: value
                    for key, value in asdict(parameter).items()
                    if value is not None and value != []
                }
                for name, parameter in self.search_space.items()
            },
        }

    @classmethod
    def from_mapping(cls, raw_config: dict[str, Any]) -> "SweepConfig":
        payload = dict(raw_config)
        base_raw = payload.get("base_config")
        if not isinstance(base_raw, dict):
            raise ValueError("Sweep config must include a 'base_config' mapping.")
        search_raw = payload.get("search_space")
        if not isinstance(search_raw, dict):
            raise ValueError("Sweep config must include a 'search_space' mapping.")
        payload["base_config"] = TrainingConfig.from_mapping(base_raw)
        payload["search_space"] = {name: parse_sweep_parameter(name, parameter) for name, parameter in search_raw.items()}
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SweepConfig":
        raw_config = load_yaml_mapping(path)
        return cls.from_mapping(raw_config)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load config files.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

        if not isinstance(raw_config, dict):
            raise ValueError(f"Config file must contain a top-level mapping: {config_path}")
    return raw_config


def dump_config_yaml(path: str | Path, config: TrainingConfig) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write config files.") from exc

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    return target


def dump_yaml_mapping(path: str | Path, payload: dict[str, Any]) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write config files.") from exc

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return target


def parse_sweep_parameter(name: str, raw_parameter: Any) -> SweepParameter:
    if isinstance(raw_parameter, dict):
        return SweepParameter(**raw_parameter)
    if isinstance(raw_parameter, (list, tuple)):
        return SweepParameter(values=list(raw_parameter))
    raise ValueError(
        f"Sweep parameter '{name}' must be defined as a mapping or list, "
        f"not {type(raw_parameter).__name__}."
    )
