#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "================================================================="
echo "   Running ETH-BQ Clinical Evaluation Benchmark Suite            "
echo "================================================================="

export PYTHONPATH="$DIR:$PYTHONPATH"
"$DIR/venv/bin/python3" "$DIR/training/evaluate.py" "$@"
