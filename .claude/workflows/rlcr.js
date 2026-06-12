export const meta = {
  name: 'rlcr',
  description: 'Iterative kernel optimization: Coder + Analyst loop until 90% roofline or stall',
  phases: [
    { title: 'Setup', detail: 'Initialize RLCR state directory and plan' },
    { title: 'Implement', detail: 'Coder writes/modifies kernel, runs correctness + benchmark' },
    { title: 'Analyze', detail: 'Analyst reviews diff, runs NCU, roofline analysis, suggests next direction' },
    { title: 'Finalize', detail: 'Write final results and conclusion' },
  ],
}

const ANALYSIS_SCHEMA = {
  type: 'object',
  properties: {
    checklist: {
      type: 'object',
      properties: {
        planAlignment: { type: 'boolean' },
        symmetry: { type: 'boolean' },
        correctness: { type: 'boolean' },
        benchmarkMethodology: { type: 'boolean' },
        codeQuality: { type: 'boolean' },
      },
      required: ['planAlignment', 'symmetry', 'correctness', 'benchmarkMethodology', 'codeQuality'],
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
    bottleneck: { type: 'string' },
    suggestions: {
      type: 'array',
      items: { type: 'string' },
    },
    rooflineEfficiency: {
      type: 'number',
      description: 'Achieved performance as percentage of theoretical peak (0-100). Based on the active bound: if memory-bound, compare achieved bandwidth vs peak bandwidth; if compute-bound, compare achieved FLOP/s vs peak FLOP/s.',
    },
    activeBound: {
      type: 'string',
      description: 'The active performance bound: compute, memory-DRAM, memory-L2, memory-L1, or latency',
    },
    assemblyInsights: {
      type: 'object',
      description: 'Low-level PTX/SASS analysis results. Only populated when assembly-level inspection was performed this round.',
      properties: {
        performed: { type: 'boolean', description: 'Whether assembly analysis was done this round' },
        registerCount: { type: 'number', description: 'Registers per thread from cuobjdump -res-usage' },
        sharedMemBytes: { type: 'number', description: 'Static shared memory in bytes' },
        keyFindings: {
          type: 'array',
          items: { type: 'string' },
          description: 'Notable SASS/PTX patterns: spills, redundant conversions, missed dual-issue, bank conflicts, etc.',
        },
      },
      required: ['performed'],
    },
    progress: { enum: ['ADVANCED', 'STALLED', 'REGRESSED'] },
    verdict: { enum: ['CONTINUE', 'COMPLETE'] },
  },
  required: ['issues', 'bottleneck', 'suggestions', 'rooflineEfficiency', 'activeBound', 'assemblyInsights', 'progress', 'verdict'],
}

const planFile = args.planFile
const baseBranch = args.baseBranch
const rlcrDir = args.rlcrDir || '.rlcr/current'
const EFFICIENCY_TARGET = 90
const STALL_LIMIT = 50

// --- Setup ---
phase('Setup')

await agent(`
Set up the RLCR state directory for this kernel optimization run.

1. Run: mkdir -p ${rlcrDir}
2. Copy the plan file ${planFile} to ${rlcrDir}/plan.md
3. Read the plan file and create ${rlcrDir}/goal-tracker.md with:
   - A "Goals" section: key milestones and acceptance criteria extracted from the plan
   - A "Progress" section: empty, will be updated each round
4. Write ${rlcrDir}/state.md with this content:
   ---
   phase: implementation
   current_round: 0
   base_branch: ${baseBranch}
   plan_file: ${planFile}
   efficiency_target: ${EFFICIENCY_TARGET}
   stall_limit: ${STALL_LIMIT}
   ---
5. Read the rules files (try ../../docs/ or docs/):
   - kernel_optimization_rules.md
   - benchmark_contract.md
   - correctness_contract.md
   Confirm they exist and report their paths.

Report what you created.
`, { label: 'setup', phase: 'Setup' })

// --- Main Loop ---
let stalledCount = 0
let round = 0

while (true) {
  // --- Coder Phase ---
  phase('Implement')
  log('Round ' + round)

  const prevAnalysisNote = round > 0
    ? 'Read the previous analysis: ' + rlcrDir + '/round-' + (round - 1) + '-analysis.md — it tells you the current bottleneck and what optimization direction to try. Fix any P0/P1 issues FIRST before new optimization work.'
    : 'This is the first round. No previous analysis exists. Start by setting up baseline/ and solution/ according to the plan.'

  await agent(
    'You are a GPU kernel optimization engineer. Implement optimizations for round ' + round + '.\n\n' +
    '## Read these files FIRST:\n' +
    '1. Plan: ' + rlcrDir + '/plan.md\n' +
    '2. Rules: find kernel_optimization_rules.md (try ../../docs/ or docs/)\n' +
    '3. Benchmark contract: find benchmark_contract.md (try ../../docs/ or docs/)\n' +
    '4. Correctness contract: find correctness_contract.md (try ../../docs/ or docs/)\n' +
    '5. ' + prevAnalysisNote + '\n\n' +
    'If external/KernelWiki/SKILL.md or external/ncu-report-skill/SKILL.md exist (resolve relative to repo root or worktree root), read them before making design decisions.\n\n' +
    '## Your Tasks:\n' +
    '1. Fix P0/P1 issues from last analysis (if any) BEFORE any new optimization\n' +
    '2. Implement the optimization direction suggested by the analyst (or the plan if round 0)\n' +
    '   - Write or modify kernel code in solution/\n' +
    '   - Ensure baseline/ has proper ABI, symmetric with candidate\n' +
    '3. Run correctness checks (python bench/correctness.py or equivalent). ALL workloads must pass.\n' +
    '4. Run benchmark (python bench/benchmark.py). Record the full output.\n' +
    '5. git add and commit your changes with a descriptive message.\n\n' +
    '## Rules (non-negotiable):\n' +
    '- Symmetric ABI between baseline and candidate — same wrapper, same build, same compile flags\n' +
    '- Correctness before performance — never skip correctness checks\n' +
    '- CUDA-event timing with A/B interleaving (use the standard benchmark template)\n' +
    '- Run nvidia-smi before and after benchmark, record GPU state\n' +
    '- Do NOT fabricate any benchmark, NCU, correctness, or GPU evidence\n' +
    '- Do NOT use --use_fast_math unless the baseline already uses it\n' +
    '- Keep all artifacts inside this task folder\n\n' +
    '## Output:\n' +
    'Write ' + rlcrDir + '/round-' + round + '-summary.md with:\n' +
    '- What was implemented or changed this round\n' +
    '- Benchmark results: per-workload median/p10/p90, geomean speedup\n' +
    '- Correctness status: pass/fail per workload\n' +
    '- Your best guess at the current bottleneck\n' +
    '- Self-assessment: is the plan complete? What is left?\n',
    { label: 'coder:r' + round, phase: 'Implement' }
  )

  // --- Analyst Phase ---
  phase('Analyze')

  const analysis = await agent(
    'You are a GPU kernel performance analyst and code reviewer for round ' + round + '.\n\n' +
    'You have TWO jobs in one pass:\n' +
    '1. REVIEW — verify the coder\'s work: correctness, methodology, code quality, plan alignment\n' +
    '2. PROFILE — run NCU profiling, do roofline analysis, identify bottlenecks, suggest next optimization directions\n\n' +
    '## Read these files:\n' +
    '1. Plan: ' + rlcrDir + '/plan.md\n' +
    '2. Coder\'s summary: ' + rlcrDir + '/round-' + round + '-summary.md\n' +
    '3. Goal tracker: ' + rlcrDir + '/goal-tracker.md\n' +
    '4. Rules: find kernel_optimization_rules.md (try ../../docs/ or docs/)\n' +
    '5. Benchmark contract: find benchmark_contract.md (try ../../docs/ or docs/)\n' +
    '6. Correctness contract: find correctness_contract.md (try ../../docs/ or docs/)\n\n' +
    'If external/ncu-report-skill/SKILL.md exists, read and follow its profiling methodology.\n\n' +
    'Run these commands:\n' +
    '  git diff ' + baseBranch + '..HEAD\n' +
    '  git log --oneline ' + baseBranch + '..HEAD\n\n' +
    '## Part 1: Review\n' +
    'Check each item:\n' +
    '- Plan Alignment: does the implementation match the plan goals?\n' +
    '- Baseline/Candidate Symmetry: same ABI, compile flags, wrapper path?\n' +
    '- Correctness: tests pass? poison-before-check? proper tolerances?\n' +
    '- Benchmark Methodology: CUDA-event timing? A/B interleaved? proper warmup/trials? inner-loop calibration? provenance?\n' +
    '- Code Quality: no dead code, debug prints, hardcoded paths, or asymmetric fast-math?\n\n' +
    '## Part 2: NCU Profiling & Roofline\n' +
    '1. Run nvidia-smi, pick an idle GPU, record model and id\n' +
    '2. Run NCU on both baseline and candidate kernels (at minimum the largest production shape)\n' +
    '   Collect: SM throughput, memory throughput, achieved occupancy, warp stall reasons, L1/L2 hit rates, DRAM bandwidth\n' +
    '3. Roofline analysis:\n' +
    '   - Calculate arithmetic intensity (FLOPs / bytes moved)\n' +
    '   - Determine compute-bound vs memory-bound\n' +
    '   - Compare achieved bandwidth or FLOP/s against hardware theoretical peak\n' +
    '   - Calculate rooflineEfficiency as a percentage: (achieved / theoretical_peak) * 100\n' +
    '   - Identify the active bound (compute, memory-DRAM, memory-L2, memory-L1, latency)\n' +
    '4. Identify the #1 bottleneck — what specific code pattern or memory access pattern causes it\n' +
    '5. Suggest 2-3 optimization directions ranked by expected benefit and risk. Be SPECIFIC.\n\n' +
    '## Part 3: Low-Level Assembly Analysis (MANDATORY every round)\n' +
    'You MUST perform PTX/SASS static analysis every round. This is not optional — every optimization iteration needs instruction-level visibility.\n\n' +
    'Commands (replace sm_100a with actual target arch):\n' +
    '- PTX (algorithm logic, instruction selection): nvcc -ptx -arch=sm_100a file.cu\n' +
    '- Cubin (binary for tools below): nvcc -cubin -arch=sm_100a file.cu\n' +
    '- SASS (real machine code — scheduling, dual-issue, registers): cuobjdump -sass file.cubin\n' +
    '- Resource usage (registers/shared mem — occupancy): cuobjdump -res-usage file.cubin\n' +
    '- Source mapping (SASS + source line numbers): nvdisasm -gi file.cubin\n' +
    '- JIT .so (FlashInfer baseline analysis): cuobjdump -sass xxx.so\n\n' +
    'What to look for:\n' +
    '- Register count vs occupancy limit — spills (STL/LDL in SASS) indicate register pressure\n' +
    '- Redundant type conversions, unnecessary MOVs, missed constant folding\n' +
    '- Instruction dependency chains blocking dual-issue\n' +
    '- Shared memory bank conflict patterns (cross-reference with NCU metrics)\n' +
    '- Compare baseline vs candidate SASS: instruction count, loop unrolling, memory access patterns\n\n' +
    'Record assembly findings in the round analysis file. assemblyInsights is a REQUIRED field in your output.\n\n' +
    '## Write your full analysis to ' + rlcrDir + '/round-' + round + '-analysis.md\n\n' +
    '## Structured output notes:\n' +
    '- rooflineEfficiency: a number 0-100 representing (achieved / theoretical_peak) * 100 for the CANDIDATE kernel on the active bound metric\n' +
    '- progress: ADVANCED if this round improved over last, STALLED if no meaningful change, REGRESSED if worse\n' +
    '- verdict: COMPLETE only if plan fully done AND correctness passes AND rooflineEfficiency >= ' + EFFICIENCY_TARGET + ' AND no P0/P1 issues. Otherwise CONTINUE.\n' +
    '- assemblyInsights: populate when you performed PTX/SASS analysis this round. Set performed=true, include registerCount, sharedMemBytes, and keyFindings.\n',
    { label: 'analyst:r' + round, phase: 'Analyze', schema: ANALYSIS_SCHEMA }
  )

  if (!analysis) {
    log('Round ' + round + ': analyst returned null — stopping')
    break
  }

  // --- Log round results ---
  const p0p1Count = analysis.issues.filter(function(i) { return i.severity === 'P0' || i.severity === 'P1' }).length
  log('Round ' + round + ': ' + analysis.progress + ' | efficiency=' + analysis.rooflineEfficiency + '% | bound=' + analysis.activeBound + ' | ' + p0p1Count + ' P0/P1')
  log('Bottleneck: ' + analysis.bottleneck)
  if (analysis.suggestions.length > 0) {
    log('Next direction: ' + analysis.suggestions[0])
  }
  if (analysis.assemblyInsights && analysis.assemblyInsights.performed) {
    var asmMsg = 'Assembly: regs=' + analysis.assemblyInsights.registerCount + ' smem=' + analysis.assemblyInsights.sharedMemBytes + 'B'
    if (analysis.assemblyInsights.keyFindings && analysis.assemblyInsights.keyFindings.length > 0) {
      asmMsg += ' | ' + analysis.assemblyInsights.keyFindings[0]
    }
    log(asmMsg)
  }

  // --- Stop condition 1: reached 90% of theoretical peak ---
  if (analysis.rooflineEfficiency >= EFFICIENCY_TARGET) {
    log('Roofline efficiency ' + analysis.rooflineEfficiency + '% >= ' + EFFICIENCY_TARGET + '% target — optimization successful!')
    phase('Finalize')
    await agent(
      'The kernel optimization has reached ' + analysis.rooflineEfficiency + '% of theoretical peak (target was ' + EFFICIENCY_TARGET + '%).\n\n' +
      'Finalize the optimization:\n' +
      '1. Ensure docs/results.md has the final per-shape performance comparison, geomean speedup, GPU info, roofline summary, and conclusion\n' +
      '2. Include the final roofline efficiency: ' + analysis.rooflineEfficiency + '% of ' + analysis.activeBound + ' peak\n' +
      '3. Verify all plan acceptance criteria are met (read ' + rlcrDir + '/plan.md)\n' +
      '4. Write ' + rlcrDir + '/complete-summary.md with the final state and all round history\n' +
      '5. Update ' + rlcrDir + '/state.md: set phase to complete\n' +
      '6. Update ' + rlcrDir + '/goal-tracker.md with final status\n' +
      '7. git add and commit final changes\n',
      { label: 'finalize', phase: 'Finalize' }
    )
    break
  }

  // --- Stop condition 2: stalled 50 consecutive rounds ---
  if (analysis.progress === 'STALLED' || analysis.progress === 'REGRESSED') {
    stalledCount++
    log('Stalled count: ' + stalledCount + '/' + STALL_LIMIT)
    if (stalledCount >= STALL_LIMIT) {
      log('No progress for ' + STALL_LIMIT + ' consecutive rounds — stopping')
      phase('Finalize')
      await agent(
        'The kernel optimization has stalled for ' + STALL_LIMIT + ' consecutive rounds.\n' +
        'Current roofline efficiency: ' + analysis.rooflineEfficiency + '% (target was ' + EFFICIENCY_TARGET + '%)\n' +
        'Active bound: ' + analysis.activeBound + '\n' +
        'Last bottleneck: ' + analysis.bottleneck + '\n\n' +
        'Write a stall report:\n' +
        '1. Write ' + rlcrDir + '/stall-summary.md with:\n' +
        '   - Rounds completed: ' + (round + 1) + '\n' +
        '   - Best achieved efficiency and the round it was achieved\n' +
        '   - Why the optimization stalled (from analyst reports)\n' +
        '   - What was tried and why it did not help\n' +
        '   - Potential directions that were NOT tried\n' +
        '2. Ensure docs/results.md has the current best per-shape performance comparison\n' +
        '3. Update ' + rlcrDir + '/state.md: set phase to stalled\n' +
        '4. Update ' + rlcrDir + '/goal-tracker.md with final status\n' +
        '5. git add and commit\n',
        { label: 'stall-report', phase: 'Finalize' }
      )
      break
    }
  } else {
    stalledCount = 0
  }

  round++
}

return {
  roundsCompleted: round + 1,
  finalEfficiency: 'see ' + rlcrDir + '/ for details',
}
