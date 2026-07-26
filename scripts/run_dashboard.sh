#!/bin/sh
set -eu

python_command=${PYTHON:-python3}
"$python_command" -m teaching_skill_miner dashboard "$@"
