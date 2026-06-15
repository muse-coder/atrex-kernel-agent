#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/launch_task.sh <kernel-task-dir> [extra claude args...]

Creates a task-owned git worktree, enters the kernel task directory inside that
worktree, and launches Claude Code with CLAUDE_PROJECT_DIR set to the kernel
directory.

Environment overrides:
  IK_BASE_BRANCH        Base branch/ref for the worktree
                        (default: current checkout branch, or HEAD if detached)
  IK_WORKTREE_BASE      Parent directory for generated worktrees
                        (default: ../IterKernel-worktrees next to this repo)
  IK_RUN_ID             Run suffix (default: timestamp-pid)
  IK_BRANCH             Exact branch name to create
  IK_BRANCH_PREFIX      Branch prefix when IK_BRANCH is unset (default: ik)
  IK_REVIEW_BASE        Exact local branch name for RLCR review base
  IK_REVIEW_BASE_PREFIX Review-base branch prefix (default: ik-base)
  IK_WORKTREE_ROOT      Exact worktree path to create
  CLAUDE_BIN            Claude executable (default: claude)
  CLAUDE_MODEL          Claude model flag value (default: opus)
  CLAUDE_EFFORT         Claude effort flag value (default: max)
  IK_BASH_BIN           Bash used for launch + spawned Claude hooks.
  IK_LAUNCHER_NAME      Friendly launcher name
  IK_TASK_LABEL         Override the friendly label for branch/worktree names
  IK_BOOTSTRAP_DRAFT=0  Skip automatic .rlcr/draft.md creation
  IK_NO_CLAUDE=1        Create the worktree without launching Claude
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

# Detect target GPU from task slug prefix.
case "$TASK_SLUG" in
  b200_*)
    TARGET_GPU_LABEL="B200"
    REMOTE_HOST_HINT="(set your B200 host)"
    ;;
  h200_*)
    TARGET_GPU_LABEL="H200"
    REMOTE_HOST_HINT="(set your H200 host)"
    ;;
  *)
    TARGET_GPU_LABEL="target"
    REMOTE_HOST_HINT="the task prompt's target host"
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
  echo "error: IterKernel requires bash 4+; /bin/bash 3.2 is not supported" >&2
  echo "hint: install modern bash and/or set IK_BASH_BIN=/path/to/bash" >&2
  exit 127
fi
IK_SELECTED_BASH_DIR="$(cd "$(dirname "$IK_SELECTED_BASH")" && pwd)"
IK_LAUNCH_PATH="$IK_SELECTED_BASH_DIR:$PATH"

if ! bash_is_safe "$BASH"; then
  if [[ "${IK_BASH_REEXECED:-}" == "1" ]]; then
    echo "error: failed to re-exec with safe bash: $IK_SELECTED_BASH" >&2
    exit 127
  fi
  export IK_BASH_REEXECED=1
  export IK_BASH_BIN="$IK_SELECTED_BASH"
  export PATH="$IK_LAUNCH_PATH"
  exec "$IK_SELECTED_BASH" "$0" "$TASK_DIR" "$@"
fi

if [[ "$TASK_DIR" = /* || "$TASK_DIR" == *".."* ]]; then
  echo "error: task dir must be repo-relative and must not contain '..': $TASK_DIR" >&2
  exit 2
fi

if [[ ! -d "$REPO_ROOT/$TASK_DIR" ]]; then
  echo "error: task dir does not exist in repo: $REPO_ROOT/$TASK_DIR" >&2
  exit 2
fi

if [[ "${IK_NO_CLAUDE:-}" != "1" ]] && ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
  echo "error: Claude executable not found: $CLAUDE_BIN" >&2
  exit 127
fi

if ! git -C "$REPO_ROOT" rev-parse --verify --quiet "$BASE_BRANCH" >/dev/null; then
  echo "error: base branch/ref not found: $BASE_BRANCH" >&2
  exit 2
fi

if ! git -C "$REPO_ROOT" cat-file -e "$BASE_BRANCH:$TASK_DIR" 2>/dev/null; then
  echo "error: base branch/ref does not contain task dir: $BASE_BRANCH:$TASK_DIR" >&2
  echo "hint: commit the kernel task folder or set IK_BASE_BRANCH" >&2
  exit 2
fi

if [[ -e "$WORKTREE_ROOT" ]]; then
  echo "error: worktree path already exists: $WORKTREE_ROOT" >&2
  echo "hint: set IK_RUN_ID or IK_WORKTREE_ROOT for a fresh path" >&2
  exit 2
fi

if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$REVIEW_BASE"; then
  echo "error: review base branch already exists: $REVIEW_BASE" >&2
  echo "hint: set IK_RUN_ID or IK_REVIEW_BASE for a fresh branch" >&2
  exit 2
fi

mkdir -p "$WORKTREE_BASE"

echo "== IterKernel task launcher =="
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
# Task Context — ${TASK_SLUG}

This file collects the task context for this kernel campaign. Open Claude Code
in this worktree and start the optimization in one command:

  /optimize-kernel <describe the kernel from the Source Prompt below>

The command runs the whole flow inline (no Workflow, no subagents) and creates
the campaign's standalone git repo under /tmp/<slug>/. See
.claude/commands/optimize-kernel.md for the full step list.

## Source Prompt

\`\`\`markdown
EOF
      cat prompt.md
      cat <<EOF
\`\`\`

## Mandatory Constraints

- Use this current kernel folder as the optimization workspace.
- Keep \`.rlcr/\` untracked.
- Read \`../../docs/benchmark_contract.md\` — its local-baseline and A/B
  benchmark rules are mandatory.
- Read \`../../docs/kernel_optimization_rules.md\` and
  \`../../docs/correctness_contract.md\`.
- Read \`${WORKTREE_ROOT}/external/KernelWiki/SKILL.md\` and
  \`${WORKTREE_ROOT}/external/ncu-report-skill/SKILL.md\` before implementation
  (when available).
- In every RLCR iteration, refresh context from the source prompt, rules,
  current benchmark/profile evidence, and knowledge skills before choosing the
  next edit.
- Recover K/R/W from the source prompt before implementation:
  - K: kernel semantics and callsite contract
  - R: correctness oracle and baseline path
  - W: workload shape set and benchmark methodology
- Check GPU state before and after every benchmark/profile run.
- Do not fabricate benchmark, NCU, correctness, or GPU-id evidence.
- Keep all artifacts inside this kernel folder.
- Keep raw profiler/NCU/build artifacts local; do not stage them for the PR.

## Recommended Approach

- Recover the baseline source, exact callsite, and workload shape set.
- Define matching baseline and candidate entry points using the same ABI.
- Fill \`bench/correctness.py\` before optimization.
- Establish \`bench/benchmark.py\`, frozen workloads, and immutable baseline numbers.
- Rank candidate directions by expected benefit and risk.
- Implement bounded optimization attempts under RLCR.
- Include a remote phase with selected host/GPU, exact commands, and artifacts.
- Use NCU/profile evidence for non-obvious bottlenecks.
- Update \`docs/results.md\` before final completion.
EOF
    } > "$DRAFT_FILE"
  fi
fi

echo
echo "== Claude project root =="
echo "$PWD"
echo
echo "Context: .rlcr/draft.md"
echo "Inside Claude Code, start the optimization in one command:"
echo "  /optimize-kernel <describe the kernel from prompt.md>"
echo

if [[ "${IK_NO_CLAUDE:-}" == "1" ]]; then
  echo "IK_NO_CLAUDE=1 set; worktree prepared without launching Claude."
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
