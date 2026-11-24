#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "================================================================="
echo "   Initiating ETH-BQ Tri-Hybrid Quantum Model Training Pipeline   "
echo "================================================================="

export PYTHONPATH="$DIR:$PYTHONPATH"
"$DIR/venv/bin/python3" "$DIR/training/train.py" "$@"
