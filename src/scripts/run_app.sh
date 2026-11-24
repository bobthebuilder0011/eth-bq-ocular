#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "================================================================="
echo "   Launching ETH-BQ Interactive Clinical Web Dashboard           "
echo "   Access via URL: http://localhost:5001                         "
echo "================================================================="

export PYTHONPATH="$DIR:$PYTHONPATH"
"$DIR/venv/bin/python3" "$DIR/web_app/app.py" "$@"
