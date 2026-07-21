#!/usr/bin/env bash

set -euo pipefail

python ipi.py validate-config --config configs/train.example.yaml
python ipi.py train --config configs/train.example.yaml
