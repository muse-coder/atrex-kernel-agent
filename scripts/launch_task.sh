#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法:
  scripts/launch_task.sh <kernel-task-dir> [额外的 claude 参数...]

创建一个任务专属的 git worktree，进入该 worktree 内的 kernel 任务目录，
并启动 Claude Code，同时把 CLAUDE_PROJECT_DIR 设为 kernel 目录。

环境变量覆盖项:
  IK_BASE_BRANCH        worktree 的基准分支/ref
                        (默认: 当前 checkout 的分支，或 detached 时为 HEAD)
  IK_WORKTREE_BASE      生成的 worktree 的父目录
                        (默认: 本 repo 旁边的 ../IterKernel-worktrees)
  IK_RUN_ID             运行后缀 (默认: timestamp-pid)
  IK_BRANCH             要创建的精确分支名
  IK_BRANCH_PREFIX      IK_BRANCH 未设置时的分支前缀 (默认: ik)
  IK_REVIEW_BASE        RLCR review base 的精确本地分支名
  IK_REVIEW_BASE_PREFIX review-base 的分支前缀 (默认: ik-base)
  IK_WORKTREE_ROOT      要创建的精确 worktree 路径
  CLAUDE_BIN            Claude 可执行文件 (默认: claude)
  CLAUDE_MODEL          Claude model flag 的值 (默认: opus)
  CLAUDE_EFFORT         Claude effort flag 的值 (默认: max)
  IK_BASH_BIN           用于启动 + 派生的 Claude hook 的 Bash。
  IK_LAUNCHER_NAME      友好的 launcher 名称
  IK_TASK_LABEL         覆盖用于分支/worktree 名称的友好标签
  IK_BOOTSTRAP_DRAFT=0  跳过自动创建 .rlcr/draft.md
  IK_NO_CLAUDE=1        只创建 worktree，不启动 Claude
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

TASK_DIR="${1%/}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
if [[ -n "${IK_BASE_BRANCH:-}" ]]; then
  BASE_BRANCH="$IK_BASE_BRANCH"
else
  BASE_BRANCH="$(
    git -C "$REPO_ROOT" symbolic-ref --quiet --short HEAD ||
      git -C "$REPO_ROOT" rev-parse --verify HEAD
  )"
fi
DEFAULT_WORKTREE_BASE="$(cd "$REPO_ROOT/.." && pwd)/IterKernel-worktrees"
WORKTREE_BASE="${IK_WORKTREE_BASE:-$DEFAULT_WORKTREE_BASE}"
RUN_ID="${IK_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
TASK_SLUG="${TASK_DIR##*/}"
LAUNCHER_NAME="${IK_LAUNCHER_NAME:-direct}"
TASK_LABEL="${IK_TASK_LABEL:-${LAUNCHER_NAME%.sh}}"
if [[ "$TASK_LABEL" == "direct" || -z "$TASK_LABEL" ]]; then
  TASK_LABEL="$TASK_SLUG"
fi
BRANCH_PREFIX="${IK_BRANCH_PREFIX:-ik}"
BRANCH="${IK_BRANCH:-${BRANCH_PREFIX}/${TASK_LABEL}-${RUN_ID}}"
REVIEW_BASE_PREFIX="${IK_REVIEW_BASE_PREFIX:-ik-base}"
REVIEW_BASE="${IK_REVIEW_BASE:-${REVIEW_BASE_PREFIX}/${TASK_LABEL}-${RUN_ID}}"
WORKTREE_ROOT="${IK_WORKTREE_ROOT:-${WORKTREE_BASE}/${TASK_LABEL}-${RUN_ID}}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CLAUDE_MODEL="${CLAUDE_MODEL:-opus}"
CLAUDE_EFFORT="${CLAUDE_EFFORT:-max}"

# 从任务 slug 前缀检测目标 GPU。
case "$TASK_SLUG" in
  b200_*)
    TARGET_GPU_LABEL="B200"
    REMOTE_HOST_HINT="(请设置你的 B200 主机)"
    ;;
  h200_*)
    TARGET_GPU_LABEL="H200"
    REMOTE_HOST_HINT="(请设置你的 H200 主机)"
    ;;
  *)
    TARGET_GPU_LABEL="target"
    REMOTE_HOST_HINT="任务 prompt 中指定的目标主机"
    ;;
esac

bash_is_safe() {
  local candidate="$1"
  [[ -n "$candidate" && -x "$candidate" ]] || return 1
  "$candidate" -c 'set -euo pipefail; a=(); : "${a[@]}"; [[ ${BASH_VERSINFO[0]} -gt 3 ]]' >/dev/null 2>&1
}

find_safe_bash() {
  local candidate
  if [[ -n "${IK_BASH_BIN:-}" ]]; then
    bash_is_safe "$IK_BASH_BIN" && printf '%s\n' "$IK_BASH_BIN"
    return
  fi
  for candidate in "$(command -v bash 2>/dev/null || true)" /opt/homebrew/bin/bash /usr/local/bin/bash; do
    if bash_is_safe "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done
}

IK_SELECTED_BASH="$(find_safe_bash || true)"
if [[ -z "$IK_SELECTED_BASH" ]]; then
  echo "错误: IterKernel 需要 bash 4+；不支持 /bin/bash 3.2" >&2
  echo "提示: 安装新版 bash，和/或设置 IK_BASH_BIN=/path/to/bash" >&2
  exit 127
fi
IK_SELECTED_BASH_DIR="$(cd "$(dirname "$IK_SELECTED_BASH")" && pwd)"
IK_LAUNCH_PATH="$IK_SELECTED_BASH_DIR:$PATH"

if ! bash_is_safe "$BASH"; then
  if [[ "${IK_BASH_REEXECED:-}" == "1" ]]; then
    echo "错误: 用安全 bash 重新 exec 失败: $IK_SELECTED_BASH" >&2
    exit 127
  fi
  export IK_BASH_REEXECED=1
  export IK_BASH_BIN="$IK_SELECTED_BASH"
  export PATH="$IK_LAUNCH_PATH"
  exec "$IK_SELECTED_BASH" "$0" "$TASK_DIR" "$@"
fi

if [[ "$TASK_DIR" = /* || "$TASK_DIR" == *".."* ]]; then
  echo "错误: 任务目录必须是 repo 相对路径，且不能包含 '..': $TASK_DIR" >&2
  exit 2
fi

if [[ ! -d "$REPO_ROOT/$TASK_DIR" ]]; then
  echo "错误: repo 中不存在该任务目录: $REPO_ROOT/$TASK_DIR" >&2
  exit 2
fi

if [[ "${IK_NO_CLAUDE:-}" != "1" ]] && ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
  echo "错误: 找不到 Claude 可执行文件: $CLAUDE_BIN" >&2
  exit 127
fi

if ! git -C "$REPO_ROOT" rev-parse --verify --quiet "$BASE_BRANCH" >/dev/null; then
  echo "错误: 找不到基准分支/ref: $BASE_BRANCH" >&2
  exit 2
fi

if ! git -C "$REPO_ROOT" cat-file -e "$BASE_BRANCH:$TASK_DIR" 2>/dev/null; then
  echo "错误: 基准分支/ref 中不包含该任务目录: $BASE_BRANCH:$TASK_DIR" >&2
  echo "提示: 提交该 kernel 任务文件夹，或设置 IK_BASE_BRANCH" >&2
  exit 2
fi

if [[ -e "$WORKTREE_ROOT" ]]; then
  echo "错误: worktree 路径已存在: $WORKTREE_ROOT" >&2
  echo "提示: 设置 IK_RUN_ID 或 IK_WORKTREE_ROOT 以使用全新路径" >&2
  exit 2
fi

if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$REVIEW_BASE"; then
  echo "错误: review base 分支已存在: $REVIEW_BASE" >&2
  echo "提示: 设置 IK_RUN_ID 或 IK_REVIEW_BASE 以使用全新分支" >&2
  exit 2
fi

mkdir -p "$WORKTREE_BASE"

echo "== IterKernel 任务启动器 =="
echo "repo:      $REPO_ROOT"
echo "launcher:  $LAUNCHER_NAME"
echo "label:     $TASK_LABEL"
echo "task:      $TASK_DIR"
echo "base:      $BASE_BRANCH"
echo "review:    $REVIEW_BASE"
echo "branch:    $BRANCH"
echo "worktree:  $WORKTREE_ROOT"
echo "bash:      $IK_SELECTED_BASH ($("$IK_SELECTED_BASH" --version | head -1))"
echo

git -C "$REPO_ROOT" branch "$REVIEW_BASE" "$BASE_BRANCH"
git -C "$REPO_ROOT" worktree add -b "$BRANCH" "$WORKTREE_ROOT" "$BASE_BRANCH"

cd "$WORKTREE_ROOT/$TASK_DIR"

if [[ "${IK_BOOTSTRAP_DRAFT:-1}" != "0" ]]; then
  mkdir -p .rlcr
  DRAFT_FILE=".rlcr/draft.md"
  if [[ ! -f "$DRAFT_FILE" ]]; then
    {
      cat <<EOF
# 任务上下文 — ${TASK_SLUG}

本文件汇总了这个 kernel campaign 的任务上下文。在该 worktree 中打开
Claude Code，用一条命令启动优化:

  /optimize-kernel <根据下面的 Source Prompt 描述这个 kernel>

该命令会由当前会话里的 master agent 编排（无 Workflow；通过 Agent 工具 spawn
analysis / code-impl / code-iter subagent），并在 /tmp/<slug>/ 下创建本 campaign
独立的 git repo。完整步骤列表见
.claude/commands/optimize-kernel.md。

## Source Prompt

\`\`\`markdown
EOF
      cat prompt.md
      cat <<EOF
\`\`\`

## 强制约束

- 使用当前这个 kernel 文件夹作为优化工作区。
- 保持 \`.rlcr/\` 不被 git 跟踪。
- 阅读 \`../../docs/benchmark_contract.md\` —— 其 local-baseline 与 A/B
  benchmark 规则是强制性的。
- 阅读 \`../../docs/kernel_optimization_rules.md\` 和
  \`../../docs/correctness_contract.md\`。
- 实现前阅读 \`${WORKTREE_ROOT}/external/KernelWiki/SKILL.md\` 和
  \`${WORKTREE_ROOT}/external/ncu-report-skill/SKILL.md\`
  （如果存在）。
- 在每一轮 RLCR 迭代中，选择下一处修改前，都要从 source prompt、规则、
  当前的 benchmark/profile 证据以及知识 skill 重新刷新 context。
- 实现前从 source prompt 中恢复 K/R/W:
  - K: kernel 语义与调用点契约
  - R: 正确性 oracle 与 baseline 路径
  - W: workload shape 集与 benchmark 方法论
- 每次 benchmark/profile 运行前后都检查 GPU 状态。
- 不要伪造 benchmark、NCU、正确性或 GPU-id 证据。
- 把所有产物都保留在这个 kernel 文件夹内。
- 把原始的 profiler/NCU/build 产物保留在本地；不要把它们 stage 进 PR。

## 推荐做法

- 恢复 baseline 源码、精确的调用点、以及 workload shape 集。
- 用相同的 ABI 定义匹配的 baseline 与 candidate 入口点。
- 优化前先填好 \`bench/correctness.py\`。
- 建立 \`bench/benchmark.py\`、冻结的 workloads、以及不可变的 baseline 数值。
- 按预期收益与风险给 candidate 方向排序。
- 在 RLCR 下实施有界的优化尝试。
- 包含一个远程阶段，写明所选 host/GPU、精确命令与产物。
- 对不明显的瓶颈使用 NCU/profile 证据。
- 最终完成前更新 \`docs/results.md\`。
EOF
    } > "$DRAFT_FILE"
  fi
fi

echo
echo "== Claude project root =="
echo "$PWD"
echo
echo "上下文: .rlcr/draft.md"
echo "在 Claude Code 中，用一条命令启动优化:"
echo "  /optimize-kernel <根据 prompt.md 描述这个 kernel>"
echo

if [[ "${IK_NO_CLAUDE:-}" == "1" ]]; then
  echo "已设置 IK_NO_CLAUDE=1；已准备好 worktree，未启动 Claude。"
  exit 0
fi

exec env \
  PATH="$IK_LAUNCH_PATH" \
  SHELL="$IK_SELECTED_BASH" \
  IK_BASH_BIN="$IK_SELECTED_BASH" \
  CLAUDE_PROJECT_DIR="$PWD" \
  "$CLAUDE_BIN" \
  --permission-mode bypassPermissions \
  --model "$CLAUDE_MODEL" \
  --effort "$CLAUDE_EFFORT" \
  "$@"
