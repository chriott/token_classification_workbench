#!/usr/bin/env bash

set -euo pipefail

token-classification validate-sweep --config configs/sweeps/sweep.example.yaml
token-classification sweep --config configs/sweeps/sweep.example.yaml
