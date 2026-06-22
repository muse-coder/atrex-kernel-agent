# NVIDIA Blackwell Ultra (B300) 硬件规格

> **来源**：用户提供的 NVIDIA **DGX B300** 系统 datasheet（8× Blackwell Ultra SXM 节点级）。
> 记录日期 2026-06-22。
>
> **数据纪律**：节点级数字为原始 datasheet 值，原样照录；**每 GPU 值为「节点值 / 8」推导**，
> 已显式标注。datasheet 未提供的项（HBM 带宽、SM 数、其他精度峰值等）**留空并注明，不臆造**。
> 用于 roofline 计算前请先补齐缺失项（尤其 HBM 带宽 = 脊点必需）。
>
> 相关：B300 属 **SM100 家族（Blackwell，SM100/SM100a；FP4-Ultra 走 SM103 变体）**，
> 与 SM120 卡（RTX PRO 系列）指令路线不同——见
> [`external/KernelWiki/wiki/hardware/sm120-hardware-specs.md`](../external/KernelWiki/wiki/hardware/sm120-hardware-specs.md)
> 的 SM100 vs SM120 对比，及
> [`external/KernelWiki/sources/docs/cutlass-changelog-sm100.md`](../external/KernelWiki/sources/docs/cutlass-changelog-sm100.md)
> 的 SM103/GB300 blockscaled FP4 支持。

## 节点级（DGX B300，8 GPU）—— 原始 datasheet 值

| 项 | 值 |
|---|---|
| GPUs | 8× NVIDIA Blackwell Ultra SXM |
| CPU | Intel® Xeon® 6776P Processors |
| Total GPU Memory | 2.1 TB |
| Performance — FP4 Tensor Core | **144 PFLOPS** \| 108 PFLOPS\* |
| Performance — FP8 Tensor Core | **72 PFLOPS**\*\* |
| Networking | 8× OSFP ports，serving 8× single-port NVIDIA® ConnectX®-8 VPI（≤ 800 Gb/s IB/Ethernet）；2× dual-port QSFP112 NVIDIA® BlueField®-3 DPU（≤ 400 Gb/s IB/Ethernet） |

> \* 与 \*\* 是原 datasheet 的脚注标记，**原文未给出脚注含义**。Blackwell 习惯上
> 「较大值含 2:4 稀疏、较小值为 dense」，但**此处不臆断**——需要精确语义时以 NVIDIA
> 官方 datasheet 脚注为准。

## 每 GPU（= 节点值 / 8，推导值，供单卡 roofline 使用）

| 项 | 每 GPU 值 | 推导 |
|---|---|---|
| GPU Memory | ≈ **262.5 GB** | 2.1 TB / 8 |
| FP4 Tensor Core | **18 PFLOPS** \| 13.5 PFLOPS\* | 144 / 8 \| 108 / 8 |
| FP8 Tensor Core | **9 PFLOPS**\*\* | 72 / 8 |

> 网络/互联（OSFP/ConnectX/BlueField）是**节点级 fabric**，与单卡 kernel roofline 无关，
> 不做 per-GPU 拆分。

## 本 datasheet 未提供（用时另查，勿臆造）

- **HBM 带宽**（roofline 脊点 = 峰值算力 / 带宽，**必需**）：本表无。
  近似参照——同属 SM100 家族的 **B200 = 8 TB/s HBM3e**（来自 KernelWiki 多处，仅作量级
  参照；B300 实际值须查 NVIDIA datasheet 再写回本文件）。
- **SM 数 / 时钟 / 功耗（TDP）**：本表无。
- **FP16 / BF16 / TF32 / INT8 峰值**：本表无（datasheet 只列了 FP4、FP8）。
- **单卡 compute capability 串**：B300 属 Blackwell Ultra；CUTLASS 以 **SM103** 支持其
  FP4-Ultra blockscaled GEMM（CUTLASS 4.2.0 起，见上文 changelog 链接）。实测目标卡时仍以
  `nvidia-smi --query-gpu=compute_cap` 为准（见 optimize-kernel.md Step 1）。

## roofline 用法提示

- 单卡 **compute-bound** GEMM/attention：用**每 GPU**峰值（FP8 9 PFLOPS / FP4 18 PFLOPS，
  含稀疏；dense 用 FP4 13.5 PFLOPS），目标 = 90% 峰值（见硬性要求 #4）。
- **脊点** = 每 GPU 峰值算力 / 每 GPU HBM 带宽 —— **HBM 带宽本表缺，必须先补齐**才能判
  compute-bound vs memory-bound，否则无法定 roofline 上限。
