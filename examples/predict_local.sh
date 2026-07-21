#!/usr/bin/env bash

set -euo pipefail

token-classification predict \
  --model-path outputs/training/baseline \
  --input-file data/prediction_input.csv
