#!/usr/bin/env bash

set -euo pipefail

python ipi.py validate-sweep --config configs/sweeps/sweep.example.yaml
python ipi.py sweep --config configs/sweeps/sweep.example.yaml
