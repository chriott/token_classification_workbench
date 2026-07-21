#!/usr/bin/env bash

set -euo pipefail

token-classification split-data \
  --input-file data/annotated_documents.jsonl \
  --output-dir data/splits \
  --seed 42 \
  --stratify-by constrained_min_labels \
  --min-label-presence 1
