# LLM Kernel-Workflow Recording Rules

These rules define what each `campaigns/llm/<model>/<platform>/` folder must
capture when running an LLM kernel-workflow campaign.

## Goal

For each priority autoregressive model: serve it on the target GPU, benchmark
at low / mid / high concurrency, profile the serving forward pass, and turn
every kernel that takes >= 1% of GPU time (excluding attention and cuDNN) into
a kernel-optimization task card.

## 1. Deployment

- Use the exact serve command from the model's upstream deployment guide.
  Record the source (doc path + commit) in `run_log.md`.
- Record the docker image, framework version/commit, host, GPU ids, and
  pre/post GPU idle state. Benchmarks measured on a non-idle GPU are invalid.

## 2. Benchmark

- Use the dataset method specified by the upstream benchmark guide.
- Sweep concurrency at three levels -- low / mid / high -- keeping the dataset
  fixed:
  - low:  `--max-concurrency 1`     (latency point)
  - mid:  `--max-concurrency 32`
  - high: `--max-concurrency 100`   (throughput point)
- Scale `--num-prompts` so each level runs long enough to be stable.
  Save the full benchmark stdout per level under `bench/`.

## 3. Profile

- Capture a torch profiler trace of the serving forward pass under
  representative load (mid concurrency is the default profiling point; profile
  high too if the kernel mix shifts).
- Keep raw traces under `profile/` (gzip them; do not stage multi-hundred-MB
  raw traces for the PR).

## 4. Kernel-workflow inventory (the deliverable)

Parse the trace to produce `docs/kernel_workflow.md` (+ `.csv`). The inventory
ranks GPU kernels by share of total GPU kernel time.

**Record** every kernel with >= 1% of GPU kernel time, except:

- **Excluded entirely** (reported only as an aggregate line, never as a task):
  - attention kernels (flash-attn / fmha / mha / mla / paged-attention)
  - cuDNN kernels (anything `cudnn*`)

**Categories** (for the kept kernels):

| Category | Examples |
|---|---|
| `gemm` | cutlass/cublas sgemm/hgemm/bf16 gemm, matmul |
| `quant_gemm` | fp8/int8/nvfp4 scaled_mm, w8a8, marlin |
| `moe` | fused_moe, grouped/group gemm, expert, topk routing |
| `norm` | rmsnorm, layernorm |
| `rope` | rotary embedding |
| `memory_bound` | elementwise, activation, add/residual, copy/cast, reduce |
| `comm` | all_reduce / all_gather / reduce_scatter / nccl |
| `other` | anything unclassified >= 1% |

Each opportunity kernel becomes a task card under `kernels/<kernel-task>/`
using the standard task format.

## 5. Cleanup

After the model's folder is committed, delete the downloaded model weights from
the remote box. Only delete weights for that model, and only after the folder
is committed.

## Layout

```text
campaigns/llm/<model>/<platform>/
  deploy.md                  # exact serve + bench + profile commands
  run_log.md                 # provenance: host, GPU ids, image, commit, idle state
  bench/                     # benchmark logs per concurrency level
  profile/                   # raw + parsed profiler artifacts
  docs/kernel_workflow.md    # the >= 1% kernel inventory
  kernels/<kernel-task>/     # per-kernel optimization task cards
```
