#!/usr/bin/env bash
# ptx_diff.sh — 归一化后的 PTX 前后对比，用于核对「我的源码改动有没有如实进 IR」
#
# PTX 判「对不对」（指令选择/降级是否符合意图），不判「快不快」（调度/spill 只在 SASS）。
# 裸 diff PTX 没用：三类与改动无关的噪声会淹没真实差异——必须先归一化：
#   1) 虚拟寄存器编号  %r12/%rd5/%f3 ...  （最大噪声源，实测降 ~8.5x）
#   2) 内部文件名哈希符号  _INTERNAL_<hash>_<n>_<file>_cu_<hash>（换文件名就全变，与改动无关）
#   3) Itanium mangling 长度前缀  _ZN39.. vs _ZN44..（改成不同长度的名字会牵动）
# 读法：看「类别对应」（改名→只动符号名 0 指令；加数组→ld/st.local；改常量→setp 立即数），
#       不要看行数（unroll + 基本块 $L__BB0_NN 重排会放大行数）。
#
# 用法:
#   scripts/ptx_diff.sh <A.ptx> <B.ptx>            # 概览 + 指令体差异
#   scripts/ptx_diff.sh <A.ptx> <B.ptx> --full     # 同时打印归一化全量 diff
# 提示: 为减少噪声，最好两版用【同一个源文件名】编译出 PTX（消掉噪声类 2/3）。
set -euo pipefail
A="${1:?usage: ptx_diff.sh <A.ptx> <B.ptx> [--full]}"
B="${2:?usage: ptx_diff.sh <A.ptx> <B.ptx> [--full]}"
MODE="${3:-}"
for f in "$A" "$B"; do [ -f "$f" ] || { echo "缺文件: $f" >&2; exit 1; }; done

norm() {  # 三类噪声归一化
  sed -E '
    /^[[:space:]]*\.loc/d;
    s/%r[0-9]+/%r/g; s/%rd[0-9]+/%rd/g; s/%f[0-9]+/%f/g; s/%fd[0-9]+/%fd/g; s/%p[0-9]+/%p/g; s/%rs[0-9]+/%rs/g;
    s/_INTERNAL_[0-9a-f]+_[0-9]+_[A-Za-z0-9_]+_cu_[0-9a-f]+/_INTERNAL_X/g;
    s/_ZN[0-9]+_INTERNAL_X/_ZN_INTERNAL_X/g;
    /^[[:space:]]*$/d' "$1"
}
# 只留「真指令行」：去掉符号声明 / 入口 / 参数 / mangled 名 行
instr_only() {
  grep -vE '\.global|\.param|\.visible|\.entry|\.extern|\.const|_INTERNAL_X|param_|_Z[0-9]+[A-Za-z]'
}

raw=$(diff "$A" "$B" | grep -cE '^[<>]' || true)
nrm=$(diff <(norm "$A") <(norm "$B") | grep -cE '^[<>]' || true)
ins=$(diff <(norm "$A") <(norm "$B") | grep -E '^[<>]' | instr_only | grep -cE '.' || true)

echo "A = $A"
echo "B = $B"
echo
printf "裸 diff 行              : %s\n" "$raw"
printf "归一化后 diff 行        : %s   (消寄存器/内部哈希/长度前缀)\n" "$nrm"
printf "其中【真指令体】差异行 : %s   <- 看这个判改动有没有进 IR\n" "$ins"
echo
echo "===== 真指令体差异（归一化, 仅指令行）====="
diff <(norm "$A") <(norm "$B") | grep -E '^[<>]' | instr_only | head -60
if [ "$ins" -gt 60 ]; then echo "... (仅显示前 60 行, 用 --full 看全量归一化 diff)"; fi

if [ "$MODE" = "--full" ]; then
  echo
  echo "===== 归一化全量 diff ====="
  diff <(norm "$A") <(norm "$B")
fi
