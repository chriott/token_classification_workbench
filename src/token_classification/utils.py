from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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
