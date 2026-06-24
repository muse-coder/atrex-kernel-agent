#!/usr/bin/env bash
# 每任务 Codex launcher 示例。为每个 kernel 任务复制并修改本文件。
# 用法: scripts/launch_tasks/example_codex.sh [额外的 codex 参数...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IK_LAUNCHER_NAME="${IK_LAUNCHER_NAME:-$(basename "$0")}"
export IK_LAUNCHER_NAME

exec "$SCRIPT_DIR/../launch_codex_task.sh" "campaigns/operators/example_task" "$@"
