#!/usr/bin/env bash

set -euo pipefail

token-classification validate-data --config configs/train.example.yaml
token-classification label-coverage --config configs/train.example.yaml
