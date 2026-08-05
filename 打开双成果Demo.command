#!/bin/zsh

# Multimodal Teaching Skill Miner: local-only dual-outcome demo launcher.
# This file is intended to be opened directly from Finder on macOS.

set -u

launcher_path="$0"
project_dir="${launcher_path:A:h}"
teachobs_root="$project_dir/artifacts/private/external_datasets/teachobs"
skill_root="$project_dir/artifacts/private/full_multimodal"

pause_on_error() {
  printf '\n启动失败：%s\n' "$1" >&2
  printf '按回车键关闭窗口…'
  IFS= read -r _
  exit 1
}

python_bin=""
python_candidates=(
  "$project_dir/.venv/bin/python"
  "/opt/homebrew/bin/python3"
  "/usr/local/bin/python3"
  "$(command -v python3 2>/dev/null || true)"
)

for candidate in "${python_candidates[@]}"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      python_bin="$candidate"
      break
    fi
  fi
done

[[ -n "$python_bin" ]] || pause_on_error "未找到 Python 3.10 或更高版本。"
[[ -d "$teachobs_root" ]] || pause_on_error "找不到真实 TeachObs 数据：$teachobs_root"
[[ -d "$skill_root" ]] || pause_on_error "找不到真实 Skill 蒸馏产物：$skill_root"

cd "$project_dir" || pause_on_error "无法进入项目目录：$project_dir"

dashboard_args=(
  -m teaching_skill_miner dashboard-real
  --teachobs-root "$teachobs_root"
  --skill-root "$skill_root"
  --initial-lesson S24
  --initial-skill linear_algebra_l03
)

if [[ "${1:-}" == "--check" ]]; then
  printf '正在自检双成果 Demo 与真实数据…\n'
  "$python_bin" "${dashboard_args[@]}" --check-data
  exit $?
fi

clear
printf '┌──────────────────────────────────────────┐\n'
printf '│  多模态教学技能挖掘 · 双成果真实 Demo       │\n'
printf '└──────────────────────────────────────────┘\n\n'
printf '正在验证并加载真实私有数据，请稍候…\n'
printf '启动后浏览器会自动打开。\n'
printf '演示结束时，请回到本窗口按 Ctrl+C。\n\n'

PYTHONUNBUFFERED=1 "$python_bin" "${dashboard_args[@]}" --port 0

exit_status=$?
if [[ $exit_status -ne 0 ]]; then
  pause_on_error "Demo 进程已异常退出（错误码 $exit_status）。"
fi

printf '\nDemo 已安全停止，可以关闭本窗口。\n'
