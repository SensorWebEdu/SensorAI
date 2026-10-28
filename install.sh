#!/usr/bin/env bash
set -e

python -m pip install -r requirements.txt
python -m pip install fastsst==0.0.4 --no-deps
