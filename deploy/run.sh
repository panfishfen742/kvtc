#!/usr/bin/env sh
set -eu

pip install -e .[dev]
pytest src/test_kvtc.py
