#!/bin/zsh
set -eu

script_dir=${0:A:h}
cd "$script_dir"

api_key_file="${DEEPSEEK_API_KEY_FILE:-$script_dir/.private/deepseek_api.txt}"
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
  print -u2 "未找到 Python 3.10 或更高版本。"
  print -u2 "请先在项目根目录创建 .venv，或安装 Python 3.10 以上版本。"
  read -k 1 "?按任意键关闭…"
  exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
  exec "$python_bin" -m teaching_skill_miner teacher-agent-dashboard --check
fi

if [[ ! -s "$api_key_file" ]]; then
  print -u2 "未找到 DeepSeek API 密钥文件。"
  print -u2 "请把密钥文件链接为：$script_dir/.private/deepseek_api.txt"
  print -u2 "也可在终端设置 DEEPSEEK_API_KEY_FILE 后启动本脚本。"
  print -u2 "密钥只由本地服务读取，不会出现在网页或日志中。"
  read -k 1 "?按任意键关闭…"
  exit 1
fi

exec "$python_bin" -m teaching_skill_miner teacher-agent-dashboard \
  --port 0 \
  --agent-backend deepseek \
  --model deepseek-v4-flash \
  --api-key-file "$api_key_file" \
  --allow-remote-student-data
