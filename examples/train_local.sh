#!/usr/bin/env bash

set -euo pipefail

token-classification validate-config --config configs/train.example.yaml
token-classification train --config configs/train.example.yaml
