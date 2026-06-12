#!/usr/bin/env bash
# Example per-task launcher. Copy and edit for each kernel task.
# Usage: scripts/launch_tasks/example.sh [extra claude args...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IK_LAUNCHER_NAME="${IK_LAUNCHER_NAME:-$(basename "$0")}"
export IK_LAUNCHER_NAME

# Point to the task directory (repo-relative path):
exec "$SCRIPT_DIR/../launch_task.sh" "campaigns/operators/example_task" "$@"
