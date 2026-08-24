#!/bin/zsh
set -eu

script_dir=${0:A:h}
cd "$script_dir"

python_bin=""
for candidate in \
  "$script_dir/.venv/bin/python" \
  "/opt/homebrew/bin/python3" \
  "/usr/local/bin/python3" \
  "$(command -v python3 2>/dev/null || true)"
do
  if [[ -n "$candidate" && -x "$candidate" ]] && \
    "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>/dev/null
  then
    python_bin="$candidate"
    break
  fi
done

if [[ -z "$python_bin" ]]; then
  print -u2 "Agent Harness 需要 Python 3.10 或更高版本。"
  read -k 1 "?按任意键关闭…"
  exit 1
fi

if [[ -z "${HARNESS_DEEPSEEK_API_KEY_FILE:-}" && -e "$script_dir/.private/deepseek_api.txt" ]]; then
  export HARNESS_DEEPSEEK_API_KEY_FILE="$script_dir/.private/deepseek_api.txt"
fi

exec "$python_bin" -m agent_harness --cwd "$script_dir" "$@"
