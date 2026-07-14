# Tile Scheduling on Blackwell

Tile scheduling combines a work-allocation mechanism with a software mapping
from work IDs to logical tensor tiles; CLC only provides the former.

**Last updated**: 2026-07-14

## Separate Three Decisions

Performance discussions often conflate three independent choices:

1. **Work allocation**: non-persistent grid, static persistent stride, software
   atomic counter, CLC cancellation/takeover, or Stream-K decomposition.
2. **ID-to-tile mapping**: linear, column-major, grouped, or swizzled raster.
3. **Tile granularity**: complete output tile, K-split, grouped-GEMM tile, or
   multi-CTA cluster tile.

CLC does not expose `init`, `query`, raster-policy, or Hilbert-policy PTX
instructions. It returns the coordinates of a canceled pending cluster. CUTLASS
or kernel code maps those coordinates to `(m, n, batch)` work.

## Work-Allocation Choices

### Non-Persistent Grid

Each CTA computes the tile associated with its launch coordinate and exits.
This is simple and often best when the grid has ample parallelism and little
runtime variance.

### Static Persistent Stride

A limited grid processes work IDs separated by `gridDim`:

```python
tile_id = program_id(0)
while tile_id < total_tiles:
    compute(tile_id)
    tile_id += num_programs(0)
```

This reduces the number of launched CTAs but does not dynamically rebalance
fixed per-worker tile sequences.

### Software Dynamic Scheduler

A resident worker obtains the next ID through a global atomic counter. This is
portable and dynamic, but adds counter initialization, memory traffic, and
contention.

### CLC Dynamic Persistent Scheduler

The launch grid represents all logical clusters. A running cluster first handles
its own `ctaid`, then asynchronously attempts to cancel another not-yet-launched
cluster. On success it processes the returned cluster coordinates. On failure it
exits and must not issue another request.

### Stream-K

Stream-K decomposes work along K to expose parallelism when complete output
tiles do not fill the machine. It solves a different problem from CLC and may
require reduction or atomic accumulation of partial tiles.

## ID-to-Tile Mapping

After obtaining a linear work ID, software can decode it using a mapping chosen
for the workload:

```python
# Linear raster
tile_m = tile_id // tiles_n
tile_n = tile_id % tiles_n

# Example grouped-M raster
group_width = group_m * tiles_n
group_id = tile_id // group_width
first_m = group_id * group_m
group_m_actual = min(tiles_m - first_m, group_m)
tile_m = first_m + (tile_id % group_m_actual)
tile_n = (tile_id % group_width) // group_m_actual
```

Swizzling can improve L2 locality, but its value depends on matrix layout,
problem shape, tile shape, and cache capacity. It is not always better than
linear order and must be benchmarked.

## CUTLASS / CuTeDSL Composition

The local SM100 CuTeDSL GEMM composes:

- `ClcDynamicPersistentTileScheduler` for CLC-backed work allocation;
- `PipelineClcFetchAsync` for overlapping the next cancellation response;
- scheduler parameters for problem shape and tile decoding; and
- separate pipelines for TMA, MMA, accumulator, and epilogue work.

The CLC response is 16 bytes per scheduler stage. Pipeline depth is a library
configuration, not a fixed hardware constant.

## Choosing a Scheduler

| Symptom | Candidate | Verify |
|---|---|---|
| Uniform large grid | Non-persistent or static persistent | launch overhead and locality |
| Variable tile duration with pending work | CLC or software atomic scheduler | runtime variance and balancing gain |
| Too few output tiles | Stream-K or smaller tiles | reduction overhead and tensor-core efficiency |
| Poor L2 hit rate | Change raster/swizzle mapping | NCU L2 metrics, not CLC status |
| Variable expert sizes | Grouped scheduler plus dynamic allocation | metadata and atomic/scheduler overhead |

## Tail Effects

CLC can reduce a tail caused by unequal worker completion times because early
finishers claim pending launches. It cannot create parallelism when the entire
grid contains fewer clusters than the hardware can run. Stream-K or smaller
tiles may be needed in that case.

For a grid of 150 one-CTA clusters, a running cluster may cancel one of the
pending launches and process its work without waiting for that launch to be
scheduled normally. The exact utilization improvement depends on residency and
runtime variance; it is not determined solely by `150 % num_sms`.

## Caveats

- Do not invent or depend on `clusterctl.init`, `clusterctl.query`, or
  `clusterctl.wait`; these are not CLC PTX instructions.
- Do not retry after a failed CLC response.
- CLC and swizzling are orthogonal; tune them independently.
- CLC request latency is workload- and implementation-dependent. Avoid
  undocumented fixed-cycle estimates.
- Multi-CTA multicast requires every CTA in the cluster to remain active through
  completion.

## Related

- [CLC Hardware Semantics](../features/clc.md)
- [Cluster Launch Control](../features/clc.md)
- [Tail Effect](tail-effect.md)
- [CUTLASS Tile Scheduling](../../common/cutedsl/cutlass-tile-scheduling.md)
- [Tile Rasterization and L2 Locality](../../common/tile-rasterization-l2-locality.md)
