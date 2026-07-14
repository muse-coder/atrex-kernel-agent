# Cluster Launch Control (CLC)

Cluster Launch Control lets a running cluster take over the coordinates of a
cluster from the same grid whose launch has not started yet.

**Last updated**: 2026-07-14

## Availability

- Introduced in PTX ISA 8.6.
- Requires `sm_100` or higher. Hopper SM90 supports Thread Block Clusters but
  not CLC.
- SM120 also satisfies the PTX target requirement. Do not conflate the absence
  of SM100 TMEM/tcgen05 features on SM120 with absence of CLC.
- The `.multicast::cluster::all` form has additional architecture-target
  requirements; consult the PTX target notes for the exact `a`/`f` target.

## Exact Hardware Semantics

CLC exposes cancellation and response-query instructions, not a programmable
tile queue:

1. A cluster begins with the work identified by its normal `%ctaid`.
2. While processing that work, one elected thread can asynchronously request
   cancellation of a cluster from the same grid that has not started yet.
3. Hardware writes a 16-byte opaque response to shared memory and completes the
   associated `mbarrier` transaction.
4. If cancellation succeeded, the response contains the CTA ID of the first CTA
   in the canceled cluster. The running cluster uses those coordinates as its
   next work item.
5. If cancellation failed, there is no pending cluster available to take over.
   After a CTA has observed failure, issuing another `try_cancel` has undefined
   behavior.

The caller does not choose which pending cluster is canceled. CLC also does not
cancel the currently running cluster or an arbitrary output tile.

## PTX Contract

The cancellation request is asynchronous:

```ptx
clusterlaunchcontrol.try_cancel.async.shared::cta
    .mbarrier::complete_tx::bytes.b128 [response_addr], [mbar_addr];
```

For a multi-CTA cluster, the response can be multicast to every CTA:

```ptx
clusterlaunchcontrol.try_cancel.async.shared::cta
    .mbarrier::complete_tx::bytes.multicast::cluster::all.b128
    [response_addr], [mbar_addr];
```

After waiting on the `mbarrier`, load the 16-byte response and query it:

```ptx
ld.shared.b128 response, [response_addr];
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, response;
@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
    {x, y, z, unused}, response;
```

Important constraints:

- `response_addr` must be naturally aligned storage for 16 bytes.
- Completion must be observed through the specified `mbarrier`; unrelated
  synchronization is not a substitute.
- With `.multicast::cluster::all`, behavior is undefined if any CTA in the
  requesting cluster has exited.
- `get_first_ctaid` is valid only after `is_canceled` reports success.

## Dynamic Work Redistribution Pattern

CLC-based scheduling normally launches a grid containing all logical work
clusters. The first item handled by each running cluster comes from `%ctaid`.
Before finishing that item, the cluster pipelines a cancellation request:

```text
current = initial %ctaid

loop:
    asynchronously try to cancel one not-yet-launched cluster
    process work identified by current
    wait for the 16-byte response

    if cancellation failed:
        exit

    current = first CTA ID of canceled cluster
    repeat
```

This produces load balancing because clusters that finish sooner can claim more
pending cluster coordinates. It is different from software persistent
scheduling, where a small resident grid advances by a fixed stride or a global
atomic counter.

## DSL and Library Mappings

### Gluon

The in-repository Gluon example maps directly to the PTX contract:

```python
mbarrier.expect(barrier, 16)
clc.try_cancel(result, barrier, multicast=True)

# Compute the current tile while the request is in flight.

mbarrier.wait(barrier, phase)
response = clc.load_result(result)
has_next = response.is_canceled()
if has_next:
    next_tile_id = response.program_id(0)
```

### CUTLASS / CuTeDSL

The local Blackwell GEMM uses
`utils.ClcDynamicPersistentTileScheduler` with
`pipeline.PipelineClcFetchAsync`. Shared storage contains one 16-byte response
per scheduler stage. A scheduler warp advances the CLC pipeline while load, MMA,
and epilogue warps process the current tile.

CLC determines which pending cluster coordinates are claimed. Mapping the
linear cluster ID to `(m, n, batch)` coordinates, raster order, and swizzling
remain scheduler-library responsibilities; they are not configurable CLC
hardware policies.

## When It Helps

CLC is most useful when:

- work duration varies between tiles or clusters;
- the grid has enough pending clusters for faster workers to take over;
- a last-wave or load-imbalance effect is visible in profiling; and
- request latency can overlap useful computation.

It is not automatically faster. A uniform workload may see little gain, and a
small grid may have no pending cluster to cancel. The repository's Gluon sample
reports one GB200 `8192^3` run at 1040.13 TFLOPS for static scheduling and
1080.74 TFLOPS with CLC (3.9%); treat this as a sample result, not a universal
speedup.

## Scheduler Choice

CLC is one persistent-scheduling model, not a replacement for every scheduler:

| Model | Grid and work progression | Best fit |
|---|---|---|
| Non-persistent | One launched CTA/cluster per tile | Large, uniform grids with enough parallelism |
| Static persistent | A resident-sized grid advances by a fixed stride | Uniform work where simple control flow wins |
| Software dynamic | Resident workers obtain IDs through a global atomic counter | Dynamic scheduling on pre-SM100 targets |
| CLC dynamic persistent | The full logical grid is launched; running clusters cancel pending launches and take their coordinates | `sm_100+` workloads with runtime variation and pending work |
| Stream-K | Multiple workers split the K dimension of output tiles | Too few complete output tiles to fill the machine |

For CLC, avoid launching only an SM-count-sized grid: that leaves no larger set
of pending cluster launches to claim. Conversely, CLC cannot create parallelism
when the complete problem contains fewer clusters than the machine can run.

## Failure Modes

- Treating failure as a transient empty-queue result and retrying violates the
  PTX contract.
- Exiting one CTA before a multicast response completes is undefined.
- Reading canceled coordinates before checking `is_canceled` is undefined.
- Attributing raster order or L2 swizzling to CLC hides the scheduler software
  that must be tuned separately.

## Common Misconceptions

| Misconception | Correct model |
|---|---|
| CLC is a hardware tile queue initialized by the kernel | The launch already defines the grid; CLC cancels one pending cluster launch and returns its coordinates |
| `try_cancel` cancels the caller | It attempts to cancel a different, not-yet-launched cluster |
| The caller supplies a tile to cancel | The caller supplies response storage and an `mbarrier`; hardware selects the pending cluster |
| CLC controls raster/swizzle policy | Software maps returned CTA IDs to logical tiles |
| Failure can be polled repeatedly | A subsequent request after observed failure is undefined |
| CLC is SM100 datacenter-only | PTX specifies `sm_100` or higher; SM120 is included, subject to qualifier-specific target rules |

## Sources

- [NVIDIA PTX ISA: `clusterlaunchcontrol.try_cancel`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
- [Gluon CLC tutorial](../../../../reference-kernels/nvidia/blackwell/gluon/12-cluster-launch-control.py)
- [Gluon multi-CTA CLC implementation](../../../../reference-kernels/nvidia/blackwell/gluon/14-multicta.py)
- [CuTeDSL dynamic persistent GEMM](../../../../reference-kernels/nvidia/blackwell/cutedsl/cutlass/dense_gemm_persistent_dynamic.py)

## Related

- [Tile Scheduling](../optimization/tile-scheduling.md)
- [Thread Block Clusters](../../common/thread-block-cluster.md)
- [mbarrier](mbarrier.md)
- [Two-SM Cooperative MMA](2sm-cooperative.md)
