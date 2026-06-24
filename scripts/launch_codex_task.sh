#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法:
  scripts/launch_codex_task.sh <kernel-task-dir> [额外的 codex 参数...]

创建一个任务专属 git worktree，并在其中启动 Codex master agent。

环境变量覆盖项:
  IK_BASE_BRANCH          worktree 的基准分支/ref（默认当前 checkout）
  IK_WORKTREE_BASE        worktree 父目录（默认 ../IterKernel-codex-worktrees）
  IK_RUN_ID               运行后缀（默认 timestamp-pid）
  IK_BRANCH               要创建的精确分支名
  IK_BRANCH_PREFIX        IK_BRANCH 未设置时的分支前缀（默认 ik-codex）
  IK_REVIEW_BASE          review base 的精确本地分支名
  IK_REVIEW_BASE_PREFIX   review-base 分支前缀（默认 ik-codex-base）
  IK_WORKTREE_ROOT        要创建的精确 worktree 路径
  CODEX_BIN               Codex 可执行文件（默认 codex）
  CODEX_MODEL             传给 codex -m；未设置则使用 Codex 默认模型
  CODEX_SANDBOX           传给 codex -s（默认 danger-full-access）
  CODEX_APPROVAL          传给 codex -a（默认 never）
  IK_LAUNCHER_NAME        友好的 launcher 名称
  IK_TASK_LABEL           覆盖用于分支/worktree 名称的友好标签
  IK_BOOTSTRAP_DRAFT=0    跳过自动创建 .rlcr/draft.md
  IK_NO_CODEX=1           只创建 worktree，不启动 Codex
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
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && git rev-parse --show-toplevel)"

if [[ -n "${IK_BASE_BRANCH:-}" ]]; then
  BASE_BRANCH="$IK_BASE_BRANCH"
else
  BASE_BRANCH="$(
    git -C "$REPO_ROOT" symbolic-ref --quiet --short HEAD ||
      git -C "$REPO_ROOT" rev-parse --verify HEAD
  )"
fi

DEFAULT_WORKTREE_BASE="$(cd "$REPO_ROOT/.." && pwd)/IterKernel-codex-worktrees"
WORKTREE_BASE="${IK_WORKTREE_BASE:-$DEFAULT_WORKTREE_BASE}"
RUN_ID="${IK_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
TASK_SLUG="${TASK_DIR##*/}"
LAUNCHER_NAME="${IK_LAUNCHER_NAME:-direct}"
TASK_LABEL="${IK_TASK_LABEL:-${LAUNCHER_NAME%.sh}}"
if [[ "$TASK_LABEL" == "direct" || -z "$TASK_LABEL" ]]; then
  TASK_LABEL="$TASK_SLUG"
fi

BRANCH_PREFIX="${IK_BRANCH_PREFIX:-ik-codex}"
BRANCH="${IK_BRANCH:-${BRANCH_PREFIX}/${TASK_LABEL}-${RUN_ID}}"
REVIEW_BASE_PREFIX="${IK_REVIEW_BASE_PREFIX:-ik-codex-base}"
REVIEW_BASE="${IK_REVIEW_BASE:-${REVIEW_BASE_PREFIX}/${TASK_LABEL}-${RUN_ID}}"
WORKTREE_ROOT="${IK_WORKTREE_ROOT:-${WORKTREE_BASE}/${TASK_LABEL}-${RUN_ID}}"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_SANDBOX="${CODEX_SANDBOX:-danger-full-access}"
CODEX_APPROVAL="${CODEX_APPROVAL:-never}"

if [[ "$TASK_DIR" = /* || "$TASK_DIR" == *".."* ]]; then
  echo "错误: 任务目录必须是 repo 相对路径，且不能包含 '..': $TASK_DIR" >&2
  exit 2
fi

if [[ ! -d "$REPO_ROOT/$TASK_DIR" ]]; then
  echo "错误: repo 中不存在该任务目录: $REPO_ROOT/$TASK_DIR" >&2
  exit 2
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

MISSING_CODEX_FILES=()
for required in \
  AGENTS.md \
  .codex/prompts/optimize-kernel.md \
  .codex/agents/master.md \
  .codex/agents/analysis.md \
  .codex/agents/code-impl.md \
  .codex/agents/code-iter.md \
  scripts/codex_round_guard.py; do
  if ! git -C "$REPO_ROOT" cat-file -e "$BASE_BRANCH:$required" 2>/dev/null; then
    MISSING_CODEX_FILES+=("$required")
  fi
done
if [[ ${#MISSING_CODEX_FILES[@]} -gt 0 ]]; then
  echo "错误: 基准分支/ref 中缺少 Codex agent 文件: $BASE_BRANCH" >&2
  printf '  - %s\n' "${MISSING_CODEX_FILES[@]}" >&2
  echo "提示: 先提交 Codex agent 文件，或设置 IK_BASE_BRANCH 指向包含它们的 ref。" >&2
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

if [[ "${IK_NO_CODEX:-}" != "1" ]] && ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
  echo "错误: 找不到 Codex 可执行文件: $CODEX_BIN" >&2
  exit 127
fi

mkdir -p "$WORKTREE_BASE"

echo "== IterKernel Codex 任务启动器 =="
echo "repo:      $REPO_ROOT"
echo "launcher:  $LAUNCHER_NAME"
echo "label:     $TASK_LABEL"
echo "task:      $TASK_DIR"
echo "base:      $BASE_BRANCH"
echo "review:    $REVIEW_BASE"
echo "branch:    $BRANCH"
echo "worktree:  $WORKTREE_ROOT"
echo "codex:     $CODEX_BIN"
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

本文件汇总了这个 kernel campaign 的任务上下文。在该 worktree 中启动 Codex 后，
Codex 应读取:

  AGENTS.md
  .codex/prompts/optimize-kernel.md

然后以 Codex master agent 流程启动优化。

## Source Prompt

\`\`\`markdown
EOF
      cat prompt.md
      cat <<EOF
\`\`\`

## Codex 强制约束

- 使用当前这个 kernel 文件夹作为优化任务来源。
- 实际 kernel campaign repo 创建在 /tmp/<slug>/。
- 保持 \`.rlcr/\` 不被 git 跟踪。
- 实现前阅读 \`docs/benchmark_contract.md\`、
  \`docs/kernel_optimization_rules.md\`、
  \`docs/correctness_contract.md\`。
- 每个 code-iter 轮读完 direction 后运行:
  \`python scripts/codex_round_guard.py mark-direction-read /tmp/<slug> <N>\`
- 每个 code-iter 轮编辑 solution 前运行:
  \`python scripts/codex_round_guard.py pre-edit /tmp/<slug> <N>\`
- 不伪造 benchmark、NCU、正确性或 GPU-id 证据。
EOF
    } >"$DRAFT_FILE"
  fi
fi

if [[ "${IK_NO_CODEX:-}" == "1" ]]; then
  echo "IK_NO_CODEX=1: 已创建 worktree，未启动 Codex。"
  exit 0
fi

cd "$WORKTREE_ROOT"

CODEX_ARGS=(-C "$WORKTREE_ROOT" -s "$CODEX_SANDBOX" -a "$CODEX_APPROVAL")
if [[ -n "${CODEX_MODEL:-}" ]]; then
  CODEX_ARGS+=(-m "$CODEX_MODEL")
fi

INITIAL_PROMPT="读取 AGENTS.md 和 .codex/prompts/optimize-kernel.md，然后以 Codex master agent 流程启动 IterKernel 优化。TASK_DIR=${TASK_DIR}。SOURCE_DRAFT=${TASK_DIR}/.rlcr/draft.md。"

exec "$CODEX_BIN" "${CODEX_ARGS[@]}" "$@" "$INITIAL_PROMPT"
