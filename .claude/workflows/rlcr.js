export const meta = {
  name: 'rlcr',
  description: 'Three-agent modular kernel optimization: Coder (CUDA) ↔ Profiler (NCU/PTX/SASS) ↔ Analyst (analysis + direction)',
  phases: [
    { title: 'Setup', detail: 'Initialize RLCR state directory and plan' },
    { title: 'Profile', detail: 'Profiler: run NCU, cuobjdump, export PTX/SASS data to files' },
    { title: 'Analyze', detail: 'Analyst: read profiling data + source → performance analysis → write direction document' },
    { title: 'Implement', detail: 'Coder: read direction document → modify CUDA code → correctness + benchmark' },
    { title: 'Finalize', detail: 'Final report with per-module contribution breakdown' },
  ],
}

// ---------------------------------------------------------------------------
// Structured output: Analyst produces this every round
// ---------------------------------------------------------------------------
var ANALYSIS_SCHEMA = {
  type: 'object',
  properties: {
    // --- phase indicator ---
    phaseCompleted: {
      enum: ['decomposition', 'module_analysis', 'integration', 'strategy_revision'],
    },

    // --- decomposition (round 0 only) ---
    decomposition: {
      type: 'object',
      properties: {
        kernelType: { type: 'string' },
        modules: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              name: { type: 'string' },
              sourceFile: { type: 'string' },
              estimatedRuntimeFraction: { type: 'number' },
              dependsOn: { type: 'array', items: { type: 'string' } },
              sharedState: { type: 'array', items: { type: 'string' } },
            },
            required: ['id', 'name', 'sourceFile', 'estimatedRuntimeFraction'],
          },
        },
        suggestedOrder: { type: 'array', items: { type: 'string' } },
      },
    },

    // --- strategy ---
    strategy: {
      type: 'object',
      properties: {
        primaryBound: { type: 'string' },
        secondaryBound: { type: 'string' },
        optimizationPhilosophy: { type: 'string' },
        phasedRoadmap: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              phase: { type: 'number' },
              targetModules: { type: 'array', items: { type: 'string' } },
              goal: { type: 'string' },
              techniques: { type: 'array', items: { type: 'string' } },
              expectedEfficiencyAfter: { type: 'number' },
            },
            required: ['phase', 'targetModules', 'goal', 'expectedEfficiencyAfter'],
          },
        },
        currentEfficiency: { type: 'number' },
      },
    },

    // --- per-round analysis ---
    theoryValidation: {
      type: 'object',
      properties: {
        predictedImprovement: { type: 'number' },
        actualImprovement: { type: 'number' },
        gapPercent: { type: 'number' },
        rootCause: { enum: ['aligned', 'implementation_gap', 'theory_error', 'both'] },
        implementationIssues: { type: 'array', items: { type: 'string' } },
        theoryCorrections: { type: 'array', items: { type: 'string' } },
      },
    },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { enum: ['P0', 'P1', 'P2', 'P3'] },
          description: { type: 'string' },
        },
        required: ['severity', 'description'],
      },
    },

    // --- common ---
    rooflineEfficiency: { type: 'number' },
    activeBound: { type: 'string' },
    bottleneck: { type: 'string' },
    moduleId: { type: 'string' },
    progress: { enum: ['ADVANCED', 'STALLED', 'REGRESSED'] },
    verdict: { enum: ['CONTINUE', 'MODULE_COMPLETE', 'MODULE_STALLED', 'STRATEGY_REVISION_NEEDED'] },
  },
  required: ['phaseCompleted', 'rooflineEfficiency', 'verdict'],
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
var planFile = args.planFile
var baseBranch = args.baseBranch
var rlcrDir = args.rlcrDir || '.rlcr/current'
var EFFICIENCY_TARGET = 90
var MODULE_ROUND_LIMIT = 15
var MODULE_STALL_LIMIT = 5

// ---------------------------------------------------------------------------
// Shared prompt fragments
// ---------------------------------------------------------------------------
var RULES_PREAMBLE =
  '## Mandatory reading (find in ../../docs/ or docs/):\n' +
  '- kernel_optimization_rules.md — CUDA C++ only, raw PTX inline asm, no CUTLASS abstractions\n' +
  '- benchmark_contract.md — CUDA-event timing, A/B interleaving, provenance\n' +
  '- correctness_contract.md — poison-before-check, NaN/Inf preservation\n\n'

var SKILL_NOTE =
  'If external/KernelWiki/SKILL.md exists, read it for architecture-specific techniques.\n' +
  'If external/ncu-report-skill/SKILL.md exists, follow its profiling methodology.\n\n'

// NCU_SKILL_PREAMBLE removed — profiler prompts now contain exact commands inline

// ============================================================
// Setup
// ============================================================
phase('Setup')

await agent(
  'Set up the RLCR state directory.\n\n' +
  '1. mkdir -p ' + rlcrDir + '/modules\n' +
  '2. mkdir -p ' + rlcrDir + '/profiles\n' +
  '3. Copy ' + planFile + ' to ' + rlcrDir + '/plan.md\n' +
  '4. Create ' + rlcrDir + '/goal-tracker.md (goals + empty progress)\n' +
  '5. Create ' + rlcrDir + '/module-tracker.json: { "modules": [], "overallBaseline": null, "completedModules": [] }\n' +
  '6. Create ' + rlcrDir + '/state.md with phase: init, base_branch: ' + baseBranch + '\n' +
  '7. Confirm rules files exist (kernel_optimization_rules.md, benchmark_contract.md, correctness_contract.md)\n\n' +
  'Report what you created.',
  { label: 'setup', phase: 'Setup' }
)

// ============================================================
// Round 0 Phase A: Profiler → Analyst — analyze baseline, design kernel architecture
// ============================================================

phase('Profile')
log('Profiler: baseline NCU + PTX/SASS export')

await agent(
  'You are a GPU profiling engineer. Read ' + rlcrDir + '/plan.md for shapes and baseline info, then run EXACTLY these commands.\n\n' +
  '## 1. Write runner\n' +
  'Write ' + rlcrDir + '/profiles/ncu_baseline_runner.py — a minimal script that:\n' +
  '  import torch; create inputs per plan.md shapes; 3x warmup; 1x call; torch.cuda.synchronize()\n\n' +
  '## 2. Discover kernel name\n' +
  'ncu --print-summary per-kernel -c 1 python ' + rlcrDir + '/profiles/ncu_baseline_runner.py\n\n' +
  '## 3. Profile (ONE command)\n' +
  'ncu --set full --section PmSampling --section PmSampling_WarpStates --section SourceCounters \\\n' +
  '  -k "regex:<KERNEL_NAME>" -c 1 -o ' + rlcrDir + '/profiles/baseline \\\n' +
  '  python ' + rlcrDir + '/profiles/ncu_baseline_runner.py\n\n' +
  '## 4. Export (ONE command)\n' +
  'ncu --import ' + rlcrDir + '/profiles/baseline.ncu-rep --page details > ' + rlcrDir + '/profiles/baseline-details.txt\n\n' +
  '## 5. Benchmark\n' +
  'python bench/benchmark.py --device cuda:0\n\n' +
  'Write ' + rlcrDir + '/profiles/baseline-manifest.md listing files created.\n' +
  'Do NOT analyze the data. Do NOT run any other ncu commands.\n',
  { label: 'profiler:baseline', phase: 'Profile' }
)

// --- Analyst: analyze baseline → design kernel architecture ---
phase('Analyze')
log('Analyst: analyze baseline → design kernel architecture + direction for initial implementation')

await agent(
  'You are a GPU kernel performance analyst. This is the INITIAL ANALYSIS phase.\n\n' +
  'The baseline (in baseline/) is a reference implementation — possibly FlashInfer, CUTLASS, or\n' +
  'another library. Your job is to deeply analyze its performance characteristics and design\n' +
  'an architecture for a NEW CUDA kernel that the Coder will implement from scratch.\n\n' +
  RULES_PREAMBLE + SKILL_NOTE +
  '## Read:\n' +
  '- Plan: ' + rlcrDir + '/plan.md\n' +
  '- If docs/module_decomposition_guide.md exists, use as reference.\n\n' +
  '## Profiler data (already exported — read these files):\n' +
  '- ' + rlcrDir + '/profiles/baseline-manifest.md\n' +
  '- ' + rlcrDir + '/profiles/baseline-res-usage.txt (if available)\n' +
  '- ' + rlcrDir + '/profiles/baseline-sass.txt (if available)\n' +
  '- ' + rlcrDir + '/profiles/baseline.ptx (if available)\n' +
  '- ' + rlcrDir + '/profiles/baseline-details.txt (NCU full details export)\n' +
  '- NCU report: ncu -i ' + rlcrDir + '/profiles/baseline.ncu-rep --csv\n\n' +
  '## Job 1: Baseline Analysis\n' +
  'Analyze the baseline kernel\'s performance:\n' +
  '- Primary bound (compute/memory/latency/barrier) — cite NCU metrics\n' +
  '- What the baseline does well (learn from it)\n' +
  '- What the baseline does poorly (opportunities)\n' +
  '- Resource utilization: registers, smem, occupancy, TC utilization\n' +
  '- Pipeline structure: how many stages, async overlap quality\n' +
  '- SASS analysis: instruction mix, spills, scheduling quality\n' +
  'Write ' + rlcrDir + '/baseline-analysis.md.\n\n' +
  '## Job 2: Kernel Architecture Design\n' +
  'Based on the baseline analysis, design the architecture for a new CUDA kernel:\n' +
  '- Tile sizes (M, N, K), CTA shape, warp layout\n' +
  '- Pipeline structure: stage count, async loading strategy\n' +
  '- Shared memory layout: buffer sizes, swizzle/padding strategy\n' +
  '- Warp specialization (if applicable): producer/consumer roles\n' +
  '- Register budget per warp/thread\n' +
  '- Key PTX instructions to use (TMA, WGMMA/UMMA/tcgen05, mbarrier, etc.)\n' +
  '- Module decomposition: identify functional modules with // MODULE: <id> markers\n' +
  '- Expected performance ceiling with derivation\n' +
  'Write ' + rlcrDir + '/kernel-architecture.md.\n\n' +
  '## Job 3: Direction Document for Coder\n' +
  'Write ' + rlcrDir + '/direction.md — this tells the Coder to implement the COMPLETE kernel.\n' +
  'Format:\n' +
  '```\n' +
  '# Initial Kernel Implementation\n\n' +
  '## Kernel Type and Semantics\n' +
  'What the kernel computes, input/output tensors, precision.\n\n' +
  '## Architecture Overview\n' +
  'Tile sizes, CTA shape, warp layout, pipeline structure (reference kernel-architecture.md).\n\n' +
  '## Module Structure\n' +
  'List all modules with // MODULE: <id> BEGIN/END markers to insert.\n' +
  'For each module: purpose, key PTX operations, register/smem budget.\n\n' +
  '## Shared Memory Layout\n' +
  'Buffer allocation, swizzle pattern, size calculation.\n\n' +
  '## Implementation Details Per Module\n' +
  'For each module, specify:\n' +
  '- Exact PTX instructions to use (with inline asm templates)\n' +
  '- Data flow: what registers/smem this module reads and writes\n' +
  '- Synchronization: which barriers, fence instructions\n\n' +
  '## Baseline Weaknesses to Exploit\n' +
  'Specific baseline problems this architecture addresses.\n\n' +
  '## Performance Target\n' +
  'Expected roofline efficiency with derivation.\n' +
  '```\n\n' +
  'git add and commit.\n\n' +
  '## Structured output:\n' +
  'phaseCompleted = "decomposition". Fill decomposition (with designed modules) + strategy.\n' +
  'Set verdict = "CONTINUE".\n',
  { label: 'analyst:architecture', phase: 'Analyze', schema: ANALYSIS_SCHEMA }
)

// ============================================================
// Round 0 Phase B: Coder implements complete kernel from scratch
// ============================================================
phase('Implement')
log('Coder: implement complete CUDA kernel based on architecture design')

await agent(
  'You are a GPU kernel optimization engineer. Your job is to implement a COMPLETE CUDA kernel\n' +
  'from scratch based on the Analyst\'s architecture design.\n\n' +
  'This is NOT incremental optimization — you are writing the full kernel for the first time.\n\n' +
  RULES_PREAMBLE +
  '## FORBIDDEN — using ANY of these is a hard failure:\n' +
  '- #include "cutlass/*.h" or #include "cute/*.hpp" (except cutlass/numeric_types.h)\n' +
  '- cutlass::gemm::collective::CollectiveBuilder\n' +
  '- cutlass::gemm::kernel::GemmUniversal\n' +
  '- cutlass::gemm::device::GemmUniversalAdapter\n' +
  '- cutlass::epilogue::collective::CollectiveBuilder\n' +
  '- using namespace cute\n' +
  '- Any CuTe layout algebra (make_layout, make_tensor, etc.)\n' +
  'If you include ANY CUTLASS/CuTe header beyond numeric_types.h, the build will be rejected.\n\n' +
  '## Reference implementation style:\n' +
  'If external/KernelWiki/SKILL.md exists, read it — especially the DeepGEMM kernel page.\n' +
  'Follow DeepGEMM style: one thin inline function per PTX instruction.\n' +
  'SM120 (Blackwell desktop) uses `mma.sync.aligned` PTX — NOT `tcgen05.mma` (SM100 datacenter only).\n\n' +
  '## Read these files FIRST:\n' +
  '- Architecture design: ' + rlcrDir + '/direction.md\n' +
  '- Detailed architecture: ' + rlcrDir + '/kernel-architecture.md\n' +
  '- Baseline analysis: ' + rlcrDir + '/baseline-analysis.md\n' +
  '- Plan: ' + rlcrDir + '/plan.md\n\n' +
  '## Implementation rules:\n' +
  '- Write CUDA C++ only — no Triton, no CuTe DSL\n' +
  '- Use raw PTX inline assembly for hardware ops (TMA, WGMMA/UMMA, mbarrier, fence)\n' +
  '- Thin wrappers only (one inline function = one PTX instruction, DeepGEMM style)\n' +
  '- Insert // MODULE: <id> BEGIN/END markers as specified in direction.md\n' +
  '- Write the kernel in solution/\n\n' +
  '## Tasks:\n' +
  '1. Implement the complete kernel following direction.md architecture.\n' +
  '2. Write the benchmark adapter (make_case, call_baseline, call_candidate).\n' +
  '3. Run python bench/correctness.py — ALL workloads must pass.\n' +
  '   If correctness fails, debug and fix until all pass.\n' +
  '4. Run python bench/benchmark.py — record full output.\n' +
  '5. git add and commit: "initial kernel implementation"\n\n' +
  '## Write summary:\n' +
  'Write ' + rlcrDir + '/initial-implementation-summary.md with:\n' +
  '- Architecture choices made and rationale\n' +
  '- Any deviations from direction.md and why\n' +
  '- Benchmark results vs baseline: per-workload, geomean speedup\n' +
  '- Correctness status\n' +
  '- Known limitations or areas for improvement\n',
  { label: 'coder:initial', phase: 'Implement' }
)

// ============================================================
// Round 0 Phase C: Profiler → Analyst on new kernel → decompose + strategy for module loop
// ============================================================
phase('Profile')
log('Profiler: profile the newly implemented kernel')

await agent(
  'You are a GPU profiling engineer. Read ' + rlcrDir + '/plan.md and config.toml for shapes/arch. Run EXACTLY these commands.\n\n' +
  '## 1. Write candidate runner\n' +
  'Write ' + rlcrDir + '/profiles/ncu_candidate_runner.py — same pattern as ncu_baseline_runner.py but calling candidate from solution/.\n\n' +
  '## 2. Discover kernel name\n' +
  'ncu --print-summary per-kernel -c 1 python ' + rlcrDir + '/profiles/ncu_candidate_runner.py\n\n' +
  '## 3. Profile (ONE command)\n' +
  'ncu --set full --section PmSampling --section PmSampling_WarpStates --section SourceCounters \\\n' +
  '  -k "regex:<KERNEL_NAME>" -c 1 -o ' + rlcrDir + '/profiles/initial \\\n' +
  '  python ' + rlcrDir + '/profiles/ncu_candidate_runner.py\n\n' +
  '## 4. Export (ONE command)\n' +
  'ncu --import ' + rlcrDir + '/profiles/initial.ncu-rep --page details > ' + rlcrDir + '/profiles/initial-details.txt\n\n' +
  '## 5. SASS/PTX dump\n' +
  'Find .cu source in solution/. Read config.toml for arch (e.g. sm_120).\n' +
  'nvcc -cubin -lineinfo -arch=sm_<ARCH> <source.cu> -o ' + rlcrDir + '/profiles/initial.cubin\n' +
  'nvcc -ptx -arch=sm_<ARCH> <source.cu> -o ' + rlcrDir + '/profiles/initial.ptx\n' +
  'cuobjdump -res-usage ' + rlcrDir + '/profiles/initial.cubin > ' + rlcrDir + '/profiles/initial-res-usage.txt\n' +
  'cuobjdump -sass ' + rlcrDir + '/profiles/initial.cubin > ' + rlcrDir + '/profiles/initial-sass.txt\n\n' +
  'Write ' + rlcrDir + '/profiles/initial-manifest.md listing files created.\n' +
  'Do NOT analyze the data. Do NOT run any other ncu commands.\n',
  { label: 'profiler:initial', phase: 'Profile' }
)

phase('Analyze')
log('Analyst: decompose new kernel + global strategy for module-level optimization')

var r0analysis = await agent(
  'You are a GPU kernel performance analyst. The Coder has implemented an initial CUDA kernel.\n' +
  'Now you need to analyze it and set up the module-level optimization loop.\n\n' +
  RULES_PREAMBLE + SKILL_NOTE +
  '## Read:\n' +
  '- Plan: ' + rlcrDir + '/plan.md\n' +
  '- Architecture design: ' + rlcrDir + '/kernel-architecture.md\n' +
  '- Coder summary: ' + rlcrDir + '/initial-implementation-summary.md\n' +
  '- If docs/module_decomposition_guide.md exists, use as reference.\n\n' +
  '## Profiler data — initial kernel (just profiled):\n' +
  '- ' + rlcrDir + '/profiles/initial-manifest.md\n' +
  '- ' + rlcrDir + '/profiles/initial-res-usage.txt\n' +
  '- ' + rlcrDir + '/profiles/initial-sass.txt\n' +
  '- ' + rlcrDir + '/profiles/initial.ptx\n' +
  '- ' + rlcrDir + '/profiles/initial-details.txt (NCU full details export)\n' +
  '- ncu -i ' + rlcrDir + '/profiles/initial.ncu-rep --csv\n\n' +
  '## Baseline profiler data (for comparison):\n' +
  '- ' + rlcrDir + '/profiles/baseline-res-usage.txt (if available)\n' +
  '- ' + rlcrDir + '/profiles/baseline-sass.txt (if available)\n' +
  '- ' + rlcrDir + '/profiles/baseline-details.txt\n' +
  '- ncu -i ' + rlcrDir + '/profiles/baseline.ncu-rep --csv\n\n' +
  '## Job 1: Verify Module Decomposition\n' +
  'Read the kernel source in solution/. Verify // MODULE: <id> BEGIN/END markers exist.\n' +
  'If missing or incorrect, insert them based on sync points, warp-role branches, functional phases.\n' +
  'Use NCU source-level data to calculate estimatedRuntimeFraction per module.\n' +
  'Write ' + rlcrDir + '/decomposition.md.\n\n' +
  '## Job 2: Gap Analysis (initial kernel vs baseline)\n' +
  'Compare the initial kernel against the baseline:\n' +
  '- Where is it faster? Where is it slower?\n' +
  '- Per-module: which modules are the biggest bottlenecks?\n' +
  '- NCU metric comparison: SM throughput, bandwidth, TC utilization, stalls\n' +
  '- PTX/SASS comparison: register count, instruction mix, spills\n\n' +
  '## Job 3: Global Optimization Strategy\n' +
  'Design the module-level optimization plan:\n' +
  '- Primary bound of the initial kernel\n' +
  '- Per-module bound type and gap to theoretical peak\n' +
  '- Optimization philosophy\n' +
  '- Phased roadmap: which modules to optimize, in what order, expected efficiency per phase\n' +
  '- Cross-module constraints\n' +
  'Write ' + rlcrDir + '/global-strategy.md.\n\n' +
  '## Job 4: First Module Direction\n' +
  'Determine the first module to optimize (highest runtime fraction or most impactful).\n' +
  'Write ' + rlcrDir + '/modules/<first-module-id>/round-0-direction.md:\n' +
  '```\n' +
  '# Optimization Direction — Module <id>, Round 0\n\n' +
  '## Target Module\n' +
  'ID, name, source file, MODULE markers location.\n\n' +
  '## Current Bottleneck\n' +
  'Bound type, specific hardware resource, NCU evidence.\n\n' +
  '## Theoretical Analysis\n' +
  'Ceiling, current gap, derivation.\n\n' +
  '## Proposed Optimization\n' +
  'What to change, why, predicted improvement % with derivation.\n\n' +
  '## Risk Assessment\n' +
  'Register pressure, smem, occupancy, cross-module impact.\n\n' +
  '## Frozen Modules\n' +
  'Modules NOT to modify.\n\n' +
  '## Acceptance Criteria\n' +
  'Target metric values.\n' +
  '```\n\n' +
  'Create a modules/<id>/ directory for each module.\n' +
  'Update module-tracker.json. git commit.\n\n' +
  '## Structured output:\n' +
  'phaseCompleted = "decomposition". Fill decomposition + strategy. Set verdict = "CONTINUE".\n',
  { label: 'analyst:decompose', phase: 'Analyze', schema: ANALYSIS_SCHEMA }
)

if (!r0analysis || !r0analysis.decomposition) {
  log('Analyst decomposition failed — cannot proceed')
  return { error: 'initial analysis failed' }
}

var modules = r0analysis.decomposition.modules
var moduleOrder = r0analysis.decomposition.suggestedOrder
var overallEfficiency = r0analysis.rooflineEfficiency || 0

log('Kernel: ' + (r0analysis.decomposition.kernelType || 'unknown'))
log('Modules: ' + modules.length)
for (var mi = 0; mi < modules.length; mi++) {
  log('  ' + modules[mi].id + ': ' + modules[mi].name + ' (' + Math.round(modules[mi].estimatedRuntimeFraction * 100) + '%)')
}
log('Order: ' + moduleOrder.join(' → '))
log('Initial kernel efficiency: ' + overallEfficiency + '%')

// ============================================================
// Module Loop — three agents rotating: Coder → Profiler → Analyst
// ============================================================
for (var modIdx = 0; modIdx < moduleOrder.length; modIdx++) {
  var moduleId = moduleOrder[modIdx]
  var moduleMeta = null
  for (var fi = 0; fi < modules.length; fi++) {
    if (modules[fi].id === moduleId) { moduleMeta = modules[fi]; break }
  }
  if (!moduleMeta) { log('Module ' + moduleId + ' not found — skipping'); continue }

  log('=== Module ' + (modIdx + 1) + '/' + moduleOrder.length + ': ' + moduleId + ' (' + moduleMeta.name + ') ===')

  var frozenModules = moduleOrder.filter(function(mid) { return mid !== moduleId })
  var moduleRound = 0
  var moduleStalledCount = 0

  while (moduleRound < MODULE_ROUND_LIMIT) {
    log('--- ' + moduleId + ' round ' + moduleRound + ' ---')

    // ========================================================
    // STEP 1: Coder — read direction.md → implement → test
    // ========================================================
    phase('Implement')

    await agent(
      'You are a GPU kernel optimization engineer. Your ONLY job is to read the direction document, implement the proposed optimization in CUDA C++, and verify correctness.\n\n' +
      RULES_PREAMBLE +
      '## FORBIDDEN — using ANY of these is a hard failure:\n' +
      '- #include "cutlass/*.h" or #include "cute/*.hpp" (except cutlass/numeric_types.h)\n' +
      '- cutlass::gemm::collective::CollectiveBuilder\n' +
      '- cutlass::gemm::kernel::GemmUniversal\n' +
      '- cutlass::gemm::device::GemmUniversalAdapter\n' +
      '- cutlass::epilogue::collective::CollectiveBuilder\n' +
      '- using namespace cute\n' +
      '- Any CuTe layout algebra (make_layout, make_tensor, etc.)\n' +
      'If you include ANY CUTLASS/CuTe header beyond numeric_types.h, the build will be rejected.\n\n' +
      '## Read the direction document FIRST:\n' +
      rlcrDir + '/modules/' + moduleId + '/round-' + moduleRound + '-direction.md\n\n' +
      'This document tells you:\n' +
      '- Which module to modify and where (MODULE markers)\n' +
      '- What the current bottleneck is\n' +
      '- What optimization to implement and why\n' +
      '- What predicted improvement to expect\n' +
      '- Which modules are frozen (do NOT touch)\n' +
      '- Acceptance criteria (target metric values)\n\n' +
      '## Also read:\n' +
      '- ' + rlcrDir + '/plan.md\n' +
      '- ' + rlcrDir + '/global-strategy.md\n' +
      (moduleRound > 0
        ? '- Previous analysis: ' + rlcrDir + '/modules/' + moduleId + '/round-' + (moduleRound - 1) + '-analysis.md — fix any P0/P1 issues FIRST.\n' +
          '- Previous direction: ' + rlcrDir + '/modules/' + moduleId + '/round-' + (moduleRound - 1) + '-direction.md — for context on what was tried.\n'
        : '') +
      '\n## Implementation rules:\n' +
      '- Write CUDA C++ only — no Triton, no CuTe DSL\n' +
      '- Use raw PTX inline assembly for hardware ops (TMA, WGMMA/UMMA, mbarrier, fence)\n' +
      '- Thin wrappers only (one inline function = one PTX instruction, DeepGEMM style)\n' +
      '- Focus on module // MODULE: ' + moduleId + ' BEGIN … END — this is the optimization target.\n' +
      '  You MAY touch code outside that module when necessary for compilation or correctness\n' +
      '  (e.g. shared data layout changes, type signature adjustments, header includes),\n' +
      '  but every change outside the target module must be a CONSEQUENCE of the module optimization,\n' +
      '  not independent refactoring. Keep outside changes minimal.\n' +
      '- Source file: ' + moduleMeta.sourceFile + '\n' +
      '- Other modules for reference (not the optimization focus): ' + frozenModules.join(', ') + '\n\n' +
      '## Tasks:\n' +
      '1. Implement the optimization described in direction.md.\n' +
      '2. Run python bench/correctness.py — ALL workloads must pass.\n' +
      '3. Run python bench/benchmark.py — record full output.\n' +
      '4. git add and commit: "' + moduleId + ' round ' + moduleRound + ': <brief description>"\n\n' +
      '## Write summary:\n' +
      'Write ' + rlcrDir + '/modules/' + moduleId + '/round-' + moduleRound + '-summary.md with:\n' +
      '- What was changed (with line references)\n' +
      '- Benchmark results: per-workload median/p10/p90, geomean speedup\n' +
      '- Correctness status: pass/fail per workload\n' +
      '- Any deviations from direction.md and why\n\n' +
      '## Do NOT run NCU or cuobjdump. The Profiler agent handles that.\n' +
      '## Do NOT do performance analysis. The Analyst agent handles that.\n',
      { label: 'coder:' + moduleId + ':r' + moduleRound, phase: 'Implement' }
    )

    // ========================================================
    // STEP 2: Profiler — run NCU + cuobjdump → export data
    // ========================================================
    phase('Profile')

    var profileDir = rlcrDir + '/profiles/' + moduleId + '-r' + moduleRound

    await agent(
      'You are a GPU profiling engineer. Run EXACTLY these commands.\n\n' +
      'mkdir -p ' + profileDir + '\n\n' +
      '## 1. Profile (ONE command)\n' +
      'ncu --set full --section PmSampling --section PmSampling_WarpStates --section SourceCounters \\\n' +
      '  -k "regex:<KERNEL_NAME>" -c 1 -o ' + profileDir + '/candidate \\\n' +
      '  python ' + rlcrDir + '/profiles/ncu_candidate_runner.py\n' +
      'Use the kernel name from previous profiler runs. If unknown, run ncu --print-summary per-kernel -c 1 first.\n\n' +
      '## 2. Export (ONE command)\n' +
      'ncu --import ' + profileDir + '/candidate.ncu-rep --page details > ' + profileDir + '/candidate-details.txt\n\n' +
      '## 3. SASS dump\n' +
      'Read config.toml for arch. Source: ' + moduleMeta.sourceFile + '\n' +
      'nvcc -cubin -lineinfo -arch=sm_<ARCH> ' + moduleMeta.sourceFile + ' -o ' + profileDir + '/candidate.cubin\n' +
      'cuobjdump -res-usage ' + profileDir + '/candidate.cubin > ' + profileDir + '/candidate-res-usage.txt\n' +
      'cuobjdump -sass ' + profileDir + '/candidate.cubin > ' + profileDir + '/candidate-sass.txt\n\n' +
      'Write ' + profileDir + '/manifest.md listing files created.\n' +
      'Do NOT analyze the data. Do NOT run any other ncu commands.\n',
      { label: 'profiler:' + moduleId + ':r' + moduleRound, phase: 'Profile' }
    )

    // ========================================================
    // STEP 3: Analyst — read profiler data + source → analyze → write direction
    // ========================================================
    phase('Analyze')

    var analysis = await agent(
      'You are a GPU kernel performance analyst for module "' + moduleId + '" round ' + moduleRound + '.\n\n' +
      'Your job: read the Profiler\'s data and the Coder\'s changes, do performance analysis, diagnose issues, and write the next direction document for the Coder.\n\n' +
      RULES_PREAMBLE + SKILL_NOTE +
      '## Read:\n' +
      '1. Global strategy: ' + rlcrDir + '/global-strategy.md\n' +
      '2. This round\'s direction: ' + rlcrDir + '/modules/' + moduleId + '/round-' + moduleRound + '-direction.md (what the Coder was told to do)\n' +
      '3. Coder summary: ' + rlcrDir + '/modules/' + moduleId + '/round-' + moduleRound + '-summary.md\n' +
      '4. Code diff: run git diff HEAD~1\n' +
      '5. Current kernel source: ' + moduleMeta.sourceFile + ' (read the FULL file, not just the diff)\n' +
      (moduleRound > 0
        ? '6. Previous analysis: ' + rlcrDir + '/modules/' + moduleId + '/round-' + (moduleRound - 1) + '-analysis.md\n'
        : '') + '\n' +
      '## Profiler data — THREE comparison levels:\n\n' +
      '### Current round (just profiled):\n' +
      '- ' + profileDir + '/manifest.md\n' +
      '- ' + profileDir + '/candidate-res-usage.txt\n' +
      '- ' + profileDir + '/candidate-sass.txt\n' +
      '- ' + profileDir + '/candidate-details.txt (NCU full details export)\n' +
      '- ncu -i ' + profileDir + '/candidate.ncu-rep --csv\n\n' +
      '### Previous round (for theory-vs-actual delta — what THIS round\'s change achieved):\n' +
      (moduleRound > 0
        ? '- ' + rlcrDir + '/profiles/' + moduleId + '-r' + (moduleRound - 1) + '/candidate-res-usage.txt\n' +
          '- ' + rlcrDir + '/profiles/' + moduleId + '-r' + (moduleRound - 1) + '/candidate-sass.txt\n' +
          '- ' + rlcrDir + '/profiles/' + moduleId + '-r' + (moduleRound - 1) + '/candidate-details.txt\n' +
          '- ncu -i ' + rlcrDir + '/profiles/' + moduleId + '-r' + (moduleRound - 1) + '/candidate.ncu-rep --csv\n\n'
        : '- (Round 0: no previous round, use original baseline below)\n\n') +
      '### Original baseline (for overall progress tracking):\n' +
      '- ' + rlcrDir + '/profiles/baseline-res-usage.txt (if available)\n' +
      '- ' + rlcrDir + '/profiles/baseline-details.txt\n' +
      '- ' + rlcrDir + '/profiles/baseline.ptx (if available)\n' +
      '- ncu -i ' + rlcrDir + '/profiles/baseline.ncu-rep --csv\n\n' +
      '## Analysis tasks:\n\n' +
      '### A) Scope Check\n' +
      'The optimization FOCUS must be module // MODULE: ' + moduleId + ' BEGIN … END.\n' +
      'Changes outside the target module are acceptable ONLY when they are a necessary\n' +
      'consequence of the module optimization (e.g. shared data layout, type signatures,\n' +
      'header includes). If the Coder made independent changes to other modules that are\n' +
      'not caused by the target module optimization → P0 issue.\n\n' +
      '### B) Theory vs Actual (compare current round vs PREVIOUS ROUND)\n' +
      'Read the predicted improvement from direction.md.\n' +
      'Compare current round benchmark against PREVIOUS ROUND benchmark (not original baseline).\n' +
      'This isolates the delta from THIS round\'s optimization.\n' +
      '- |gap| < 20% → rootCause = "aligned"\n' +
      '- actual < predicted → check:\n' +
      '  IMPLEMENTATION GAP: compare res-usage (spills?), SASS (instruction count, STL/LDL, unrolling), NCU (bank conflicts, stall changes)\n' +
      '  THEORY ERROR: bottleneck type changed? ceiling estimate wrong? Amdahl effect?\n' +
      '- actual > predicted → understand why, update model\n\n' +
      '### C) PTX/SASS Deep Analysis (compare current vs previous round)\n' +
      '- Register count delta and spill analysis (STL/LDL in SASS)\n' +
      '- Instruction count and loop structure changes\n' +
      '- Dual-issue patterns, dependency chains\n' +
      '- Shared memory access patterns (bank conflicts)\n\n' +
      '### D) NCU Metrics Analysis (compare current vs previous round + vs original baseline)\n' +
      '- SM throughput, DRAM bandwidth, L2/L1 hit rates\n' +
      '- Warp stall reasons distribution (current vs previous vs baseline)\n' +
      '- Tensor Core utilization\n' +
      '- Achieved occupancy\n' +
      '- Overall progress: current efficiency vs original baseline efficiency\n\n' +
      '### E) Strategy Trajectory Check\n' +
      'Does current efficiency match the phasedRoadmap prediction?\n' +
      'If deviation > 10% → verdict = "STRATEGY_REVISION_NEEDED"\n\n' +
      '## Write analysis:\n' +
      'Write ' + rlcrDir + '/modules/' + moduleId + '/round-' + moduleRound + '-analysis.md with FULL diagnosis.\n' +
      'Include: all NCU metric comparisons, PTX/SASS findings, theory validation, identified issues.\n\n' +
      '## Write next direction document:\n' +
      'If verdict is CONTINUE, write ' + rlcrDir + '/modules/' + moduleId + '/round-' + (moduleRound + 1) + '-direction.md:\n' +
      '```\n' +
      '# Optimization Direction — Module ' + moduleId + ', Round ' + (moduleRound + 1) + '\n\n' +
      '## Problems Found This Round\n' +
      '[List specific issues from the analysis with NCU/SASS evidence]\n\n' +
      '## Target Module\n' +
      'ID, source file, MODULE markers.\n\n' +
      '## Current Bottleneck\n' +
      'Bound type, specific resource, NCU metric values.\n\n' +
      '## Theoretical Analysis\n' +
      'Ceiling, current gap, derivation.\n\n' +
      '## Proposed Optimization\n' +
      'What to change, why (PTX-level reasoning), predicted improvement % with math.\n\n' +
      '## Risk Assessment\n' +
      'Register pressure, smem, occupancy.\n\n' +
      '## Frozen Modules\n' +
      frozenModules.join(', ') + '\n\n' +
      '## Acceptance Criteria\n' +
      'Target metric values.\n' +
      '```\n\n' +
      '## Verdict:\n' +
      '- CONTINUE: more optimization potential, direction.md written\n' +
      '- MODULE_COMPLETE: reached ceiling or diminishing returns\n' +
      '- MODULE_STALLED: no progress, corrections exhausted\n' +
      '- STRATEGY_REVISION_NEEDED: roadmap trajectory diverged\n',
      { label: 'analyst:' + moduleId + ':r' + moduleRound, phase: 'Analyze', schema: ANALYSIS_SCHEMA }
    )

    if (!analysis) {
      log('Analyst returned null — stopping module ' + moduleId)
      break
    }

    // --- Log ---
    var tv = analysis.theoryValidation
    if (tv) {
      log('Theory: predicted=' + tv.predictedImprovement + '% actual=' + tv.actualImprovement + '% cause=' + tv.rootCause)
    }
    log('Progress: ' + (analysis.progress || '-') + ' | Efficiency: ' + analysis.rooflineEfficiency + '% | Verdict: ' + analysis.verdict)

    overallEfficiency = analysis.rooflineEfficiency

    // --- Strategy revision ---
    if (analysis.verdict === 'STRATEGY_REVISION_NEEDED') {
      log('Trajectory diverged — revising global strategy')
      phase('Analyze')

      var revision = await agent(
        'You are a GPU kernel performance analyst. The global strategy needs REVISION.\n\n' +
        RULES_PREAMBLE + SKILL_NOTE +
        'Current efficiency: ' + overallEfficiency + '%\n\n' +
        '## Read:\n' +
        '- ' + rlcrDir + '/global-strategy.md\n' +
        '- ' + rlcrDir + '/module-tracker.json\n' +
        '- All analysis files in ' + rlcrDir + '/modules/\n' +
        '- Latest profiler data in ' + profileDir + '/\n\n' +
        '## Tasks:\n' +
        '1. Re-analyze bottleneck landscape from latest NCU data.\n' +
        '2. Diagnose why strategy failed.\n' +
        '3. Update optimizationPhilosophy, rebuild phasedRoadmap for remaining modules.\n' +
        '4. Overwrite ' + rlcrDir + '/global-strategy.md (add Revision History).\n' +
        '5. Write revised direction: ' + rlcrDir + '/modules/' + moduleId + '/round-' + (moduleRound + 1) + '-direction.md.\n' +
        '6. git commit: "revise strategy after ' + moduleId + '"\n\n' +
        'phaseCompleted = "strategy_revision". verdict = "CONTINUE".\n',
        { label: 'analyst:strategy-rev:' + moduleId, phase: 'Analyze', schema: ANALYSIS_SCHEMA }
      )

      if (revision && revision.strategy && revision.strategy.phasedRoadmap) {
        var newOrder = []
        for (var nri = 0; nri < revision.strategy.phasedRoadmap.length; nri++) {
          var nrp = revision.strategy.phasedRoadmap[nri]
          for (var nti = 0; nti < nrp.targetModules.length; nti++) {
            var nmid = nrp.targetModules[nti]
            var alreadyDone = false
            for (var di = 0; di <= modIdx; di++) {
              if (moduleOrder[di] === nmid) { alreadyDone = true; break }
            }
            if (!alreadyDone && newOrder.indexOf(nmid) === -1) newOrder.push(nmid)
          }
        }
        if (newOrder.length > 0) {
          for (var si = 0; si < newOrder.length && (modIdx + 1 + si) < moduleOrder.length; si++) {
            moduleOrder[modIdx + 1 + si] = newOrder[si]
          }
          log('Module order updated: ' + moduleOrder.slice(modIdx + 1).join(' → '))
        }
      }
      moduleRound++
      continue
    }

    // --- Target reached ---
    if (overallEfficiency >= EFFICIENCY_TARGET) {
      log('Efficiency ' + overallEfficiency + '% >= target — done!')
      break
    }

    // --- Module complete ---
    if (analysis.verdict === 'MODULE_COMPLETE') {
      log('Module ' + moduleId + ': COMPLETE')
      break
    }

    // --- Module stalled ---
    if (analysis.verdict === 'MODULE_STALLED' || analysis.progress === 'STALLED') {
      moduleStalledCount++
      if (moduleStalledCount >= MODULE_STALL_LIMIT) {
        log('Module ' + moduleId + ': stalled ' + MODULE_STALL_LIMIT + ' rounds — moving on')
        break
      }
    } else {
      moduleStalledCount = 0
    }

    moduleRound++
  }

  // --- Integration after each module ---
  if (moduleRound > 0) {
    phase('Profile')
    log('Integration profiling for module ' + moduleId)

    var integrationDir = rlcrDir + '/profiles/' + moduleId + '-integration'

    await agent(
      'You are a GPU profiling engineer. Run EXACTLY these commands.\n\n' +
      'mkdir -p ' + integrationDir + '\n\n' +
      '## 1. Profile (ONE command)\n' +
      'ncu --set full --section PmSampling --section PmSampling_WarpStates --section SourceCounters \\\n' +
      '  -k "regex:<KERNEL_NAME>" -c 1 -o ' + integrationDir + '/integration \\\n' +
      '  python ' + rlcrDir + '/profiles/ncu_candidate_runner.py\n' +
      'Use the kernel name from previous profiler runs. If unknown, run ncu --print-summary per-kernel -c 1 first.\n\n' +
      '## 2. Export (ONE command)\n' +
      'ncu --import ' + integrationDir + '/integration.ncu-rep --page details > ' + integrationDir + '/integration-details.txt\n\n' +
      '## 3. SASS/PTX dump\n' +
      'Read config.toml for arch. Source: ' + moduleMeta.sourceFile + '\n' +
      'nvcc -cubin -lineinfo -arch=sm_<ARCH> ' + moduleMeta.sourceFile + ' -o ' + integrationDir + '/integration.cubin\n' +
      'nvcc -ptx -arch=sm_<ARCH> ' + moduleMeta.sourceFile + ' -o ' + integrationDir + '/integration.ptx\n' +
      'cuobjdump -res-usage ' + integrationDir + '/integration.cubin > ' + integrationDir + '/integration-res-usage.txt\n' +
      'cuobjdump -sass ' + integrationDir + '/integration.cubin > ' + integrationDir + '/integration-sass.txt\n\n' +
      '## 4. Correctness + benchmark\n' +
      'python bench/correctness.py\n' +
      'python bench/benchmark.py --device cuda:0\n\n' +
      'Write ' + integrationDir + '/manifest.md listing files created.\n' +
      'Do NOT analyze the data. Do NOT run any other ncu commands.\n',
      { label: 'profiler:integrate:' + moduleId, phase: 'Profile' }
    )

    phase('Analyze')
    log('Integration analysis for module ' + moduleId)

    await agent(
      'You are a GPU kernel performance analyst. Module "' + moduleId + '" completed ' + moduleRound + ' rounds.\n\n' +
      RULES_PREAMBLE +
      '## Read:\n' +
      '- ' + rlcrDir + '/global-strategy.md\n' +
      '- ' + rlcrDir + '/module-tracker.json\n' +
      '- Kernel source: ' + moduleMeta.sourceFile + ' (read the FULL file)\n' +
      '- All analysis files: ' + rlcrDir + '/modules/' + moduleId + '/round-*-analysis.md\n\n' +
      '## Integration profiler data:\n' +
      '- ' + integrationDir + '/manifest.md\n' +
      '- ' + integrationDir + '/integration-res-usage.txt\n' +
      '- ' + integrationDir + '/integration-sass.txt\n' +
      '- ' + integrationDir + '/integration.ptx\n' +
      '- ' + integrationDir + '/integration-details.txt (NCU full details export)\n' +
      '- ncu -i ' + integrationDir + '/integration.ncu-rep --csv\n\n' +
      '## Baseline data (for overall comparison):\n' +
      '- ' + rlcrDir + '/profiles/baseline-res-usage.txt (if available)\n' +
      '- ' + rlcrDir + '/profiles/baseline-details.txt\n' +
      '- ncu -i ' + rlcrDir + '/profiles/baseline.ncu-rep --csv\n\n' +
      '## Tasks:\n' +
      '1. Compare full-kernel performance against baseline.\n' +
      '2. If REGRESSED: read NCU + SASS data → diagnose (compiler regression / resource conflict / pipeline bubble / bad theory)\n' +
      '   Write ' + rlcrDir + '/modules/' + moduleId + '/regression-analysis.md\n' +
      '3. Update ' + rlcrDir + '/module-tracker.json: status, rounds, contribution.\n' +
      '4. Update ' + rlcrDir + '/goal-tracker.md.\n' +
      (modIdx + 1 < moduleOrder.length
        ? '5. Write direction for next module: ' + rlcrDir + '/modules/' + moduleOrder[modIdx + 1] + '/round-0-direction.md\n'
        : '5. No more modules — skip direction writing.\n') +
      '6. git commit.\n\n' +
      'phaseCompleted = "integration". Fill rooflineEfficiency.\n',
      { label: 'analyst:integrate:' + moduleId, phase: 'Analyze', schema: ANALYSIS_SCHEMA }
    )
  }

  if (overallEfficiency >= EFFICIENCY_TARGET) break
}

// ============================================================
// Finalize
// ============================================================
phase('Finalize')

await agent(
  'Modular kernel optimization complete.\n' +
  'Overall efficiency: ' + overallEfficiency + '% (target: ' + EFFICIENCY_TARGET + '%)\n' +
  'Modules: ' + moduleOrder.join(', ') + '\n\n' +
  '## Write final report:\n' +
  '1. Read ' + rlcrDir + '/module-tracker.json\n' +
  '2. Write docs/results.md:\n' +
  '   - Per-module contribution breakdown\n' +
  '   - Theory accuracy summary\n' +
  '   - Final per-shape performance, geomean speedup\n' +
  '   - GPU info, roofline summary\n' +
  '3. Write ' + rlcrDir + '/complete-summary.md\n' +
  '4. Update ' + rlcrDir + '/state.md: phase = ' + (overallEfficiency >= EFFICIENCY_TARGET ? 'complete' : 'stalled') + '\n' +
  '5. git commit.\n',
  { label: 'finalize', phase: 'Finalize' }
)

return {
  modulesOptimized: moduleOrder.length,
  finalEfficiency: overallEfficiency,
  target: EFFICIENCY_TARGET,
  status: overallEfficiency >= EFFICIENCY_TARGET ? 'target_reached' : 'campaign_complete',
}
