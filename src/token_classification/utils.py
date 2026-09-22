from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_timestamped_run_directory(
    output_dir: str | Path,
    name: str,
    *,
    timestamp: str | None = None,
) -> Path:
    """Create a unique run directory using the host system's local time."""
    parent = ensure_directory(output_dir)
    timestamp = timestamp or datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"{name}_{timestamp}"
    candidate = parent / base_name
    suffix = 2
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = parent / f"{base_name}_{suffix:02d}"
            suffix += 1


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target


def remove_checkpoint_directories(run_output_dir: str | Path) -> None:
    run_output_dir = Path(run_output_dir)
    for checkpoint_dir in run_output_dir.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir)


def remove_model_weight_files(run_output_dir: str | Path) -> None:
    run_output_dir = Path(run_output_dir)
    for pattern in ("model*.safetensors", "pytorch_model*.bin"):
        for weight_file in run_output_dir.glob(pattern):
            if weight_file.is_file():
                weight_file.unlink()
