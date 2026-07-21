#!/usr/bin/env bash

set -euo pipefail

python ipi.py validate-data --config configs/train.example.yaml
python ipi.py label-coverage --config configs/train.example.yaml
