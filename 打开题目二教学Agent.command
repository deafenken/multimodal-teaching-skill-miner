#!/bin/zsh
set -eu

script_dir=${0:A:h}
cd "$script_dir"

# The macOS double-click entry point opens the local Console after it is ready.
# Keep this overrideable for terminal/headless use: TEACHLAB_OPEN_BROWSER=0 disables it.
: "${TEACHLAB_OPEN_BROWSER:=1}"
export TEACHLAB_OPEN_BROWSER

api_key_file="${DEEPSEEK_API_KEY_FILE:-$script_dir/.private/deepseek_api.txt}"
python_bin=""
node_bin=""
npm_bin=""

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

for candidate in \
  "/opt/homebrew/bin/node" \
  "/usr/local/bin/node" \
  "$(command -v node 2>/dev/null || true)"
do
  if [[ -n "$candidate" && -x "$candidate" ]] && \
    "$candidate" -e 'process.exit(Number(process.versions.node.split(".")[0]) < 22 ? 1 : 0)' 2>/dev/null
  then
    node_bin="$candidate"
    npm_candidate="${candidate:h}/npm"
    if [[ -x "$npm_candidate" ]]; then
      npm_bin="$npm_candidate"
    else
      npm_bin="$(command -v npm 2>/dev/null || true)"
    fi
    break
  fi
done

if [[ "${1:-}" == "--stop" ]]; then
  if [[ -z "$node_bin" ]]; then
    print -u2 "停止 Console 需要 Node.js 22+。"
    exit 1
  fi
  exec "$node_bin" "$script_dir/scripts/start_teacher_agent_console.mjs" --stop
fi

if [[ -z "$python_bin" ]]; then
  print -u2 "未找到 Python 3.10 或更高版本。"
  print -u2 "请先在项目根目录创建 .venv，或安装 Python 3.10 以上版本。"
  read -k 1 "?按任意键关闭…"
  exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
  exec "$python_bin" -m teaching_skill_miner teacher-agent-dashboard --check
fi

if [[ -z "$node_bin" || -z "$npm_bin" ]]; then
  print -u2 "未找到 Node.js 22+ 与 npm。"
  print -u2 "新 Console 需要 Node.js 22 或更高版本。"
  read -k 1 "?按任意键关闭…"
  exit 1
fi

if [[ ! -d "$script_dir/apps/console/node_modules" ]]; then
  print -u2 "新 Console 依赖尚未安装。"
  print -u2 "请先执行：cd '$script_dir/apps/console' && npm ci"
  read -k 1 "?按任意键关闭…"
  exit 1
fi

mkdir -p "$script_dir/.private"
chmod 700 "$script_dir/.private"
exec "$node_bin" "$script_dir/scripts/start_teacher_agent_console.mjs" --daemon "$python_bin" "$api_key_file" "$npm_bin"
