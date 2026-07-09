# π0.5 bs=1 inference — minimal reproduction guide

Everything needed to reproduce our π0.5 single-sample inference latency
numbers and start kernel-level optimization, without touching Ray, the env
stack, or a robot. One script (`standalone_infer_bench.py`), one checkpoint,
one container.

## Target numbers (the baseline you should reproduce)

Measured on **RTX PRO 5000 Blackwell (sm_120, 110 SMs, 72 GB)**, bs=1,
`num_steps=10` denoise steps, 3 cameras, `torch.compile max-autotune`
(the production path, #968). All software optimizations are already in this
branch + the openpi patches:

| metric | value | note |
|---|---|---|
| e2e `predict_action_batch` (CPU wall clock) | **58.9 ms** | headline number |
| GPU span of `sample_actions` | 55.5 ms | nsys projection (span, not busy) |
| `prefix/vlm_forward` | 23.5 ms | MFU ~71 %, compute-bound — no headroom |
| `prefix/vision_siglip` (batched 3 views) | 6.4 ms | MFU 43 % |
| `denoise/expert_forward` × 10 steps | 2.21 ms/step | **MFU 7.4 %, MBU ~25 % — kernel target** |
| measured device peaks (same box) | 239 TFLOPS bf16, 1108 GB/s D2D | used for MFU/MBU above |

**Cross-GPU caveat:** absolute milliseconds are anchored to the RTX PRO 5000.
On other parts (H100/H20) the *structure* — phase ranking, MFU conclusions,
the denoise expert being occupancy-bound — holds, but do not compare raw ms
(H20 reference: ~73 ms e2e compiled).

## The optimization target

The VLM prefill is compute-bound at ~71 % MFU; leave it alone. The money is
in the **denoise loop**: 10 sequential calls into the 300M Gemma action
expert, each a chain of skinny GEMMs (M = 51 tokens × bs). At bs=1 they run
at 7.4 % MFU / ~25 % MBU — occupancy-bound, not bandwidth-bound. Directions
we consider open: skinny-GEMM kernels tuned for M≈51, weight quantization
(the expert is read 10× per predict), fusion across the per-step chain.
Batching is **not** an option: this is a real-robot control loop, one
observation at a time (locked decision).

## Environment

- **Image:** `docker.io/chenchaoxnv/rlinf:0.2-maniskill_libero-blackwell`
  (CUDA 12.8.1, torch 2.7.1+cu128, nsys 2025.3.1, flash-attn for
  sm_90/100/120). The openpi environment ships in the image at
  `/opt/venv/openpi` — activate it before running:
  `source /opt/venv/openpi/bin/activate`. The patches in step 2 below are
  based on that openpi build (2026-07); if you use a different openpi
  install instead, diff before applying.
- **GPU:** anything sm_90+; see the cross-GPU caveat above.

### 1. Repo

```bash
git clone -b feature/pi05-rollout-profiling-nvtx https://github.com/chenchaoxu7575/RLinf.git
cd RLinf
```

### 2. openpi patches (required — part of the measured baseline)

Three openpi files must be replaced *inside the openpi install* (batched
SigLIP call, `@at.typecheck` no-op, fine-grained NVTX). Without them you
will measure ~66 ms instead of ~59 ms and lose the NVTX phase split:

```bash
patches/openpi/apply.sh                       # default: /opt/venv/openpi/...
patches/openpi/apply.sh /path/to/site-packages   # custom venv
```

See `patches/openpi/README.md` for what each patch does and why the
typecheck gate must live at its definition site.

### 3. Checkpoint

One download contains the safetensors and the norm stats:

```bash
hf download RLinf/RLinf-Pi05-LIBERO-SFT \
    --local-dir /workspace/rlinf_pub/models/RLinf-Pi05-LIBERO-SFT
```

(`hf` is the current huggingface_hub CLI; the legacy `huggingface-cli`
entry point is deprecated and non-functional in recent versions.)

First run also fetches the PaliGemma tokenizer into `~/.cache/openpi/`;
on an offline box, copy that cache dir from a machine that has it.

## Run

```bash
python benchmarks/pi05_infer/standalone_infer_bench.py \
    --model-path /workspace/rlinf_pub/models/RLinf-Pi05-LIBERO-SFT
```

Defaults reproduce the baseline config exactly: `pi05_turtle`, bs=1,
10 denoise steps, action chunk 50, action dim 6, 3× 128×128 cameras
(resized to 224 inside the transform, as in production), max-autotune.

- **Warmup takes minutes** on the first run: max-autotune autotunes GEMMs
  and captures an inductor CUDA graph for the expert. Timed iterations only
  start after warmup. Expect `cpu wall clock mean ≈ 58.9 ms` on the
  reference box.
- `--phases` prints a sync-timed decomposition (obs/tokenize CPU work,
  VLM prefill, denoise loop, output transform) matching the NVTX taxonomy.
- `--no-compile` gives the eager baseline (useful to sanity-check kernels
  outside the inductor graph, but production is compiled).
- `--iters/--warmup/--batch-size` for sweeps.

## Profiling with nsys (same caliber as our reps)

```bash
nsys profile -t cuda,cudnn,cublas,nvtx --sample=none \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node \
    -o pi05_infer \
    python benchmarks/pi05_infer/standalone_infer_bench.py --cuda-profiler \
        --model-path /workspace/rlinf_pub/models/RLinf-Pi05-LIBERO-SFT
```

- `--cuda-profiler` gates capture via `torch.cuda.profiler.start()/stop()`
  so the rep contains only steady-state iterations (no compile noise).
- `--cuda-graph-trace=node` is essential: the denoise expert runs inside an
  inductor CUDA graph; the default "graph" mode shows one opaque blob.
- NVTX taxonomy (all already in the model code + patches):
  `predict/*` (obs → transform → sample_actions → output),
  `prefix/*` (`vision_siglip`, `lang_embed`, `mask_prep`, `vlm_forward`),
  `denoise/*` (`preprocess`, `prefix_cache`, `loop`, `step`),
  `bench/iterN` from this script.
- When reading the rep: nsys "GPU projection" rows are **spans**, not busy
  time. Use the e2e wall-clock row for MFU math; per-phase rows are for
  locating work, not for utilization claims.

### GPU metrics sampling (SM occupancy rows) in containers

Perf-counter access needs admin in the *init* user namespace
(`RmProfilingAdminOnly=1`). Under Docker, run privileged. Under enroot, a
plain `enroot start` creates a user namespace and fails with
`ERR_NVGPUCTRPERM` — surfaced misleadingly as
`Illegal --gpu-metrics-devices argument`. Working enroot recipe:

```bash
systemctl stop dcgm_exporter          # if present — it holds the counters; restart after
enroot start --rw --mount ... <container> sleep infinity &   # sleeper owns the mount ns
nsenter -t <sleeper-pid> -m bash -c 'cd /workspace/... && nsys profile --gpu-metrics-devices=0 ...'
```

Then add `--gpu-metrics-devices=<system gpu index>` to the nsys command.

## Known pitfalls

1. **Recompile on shape change:** the first predict after any input-shape
   change (batch size, image count) triggers a full recompile. Never time
   without warmup at the final shapes (the script handles this).
2. **Typecheck patch is import-order sensitive:** gating `@at.typecheck`
   anywhere but its definition site in `shared/array_typing.py` silently
   does nothing (dataconfig imports decorate the classes first). Use
   `apply.sh`, don't hand-roll.
3. **Don't compare per-phase sums to e2e** to the last ms: the production
   path overlaps CPU transform work in a thread pool; the `--phases` mode
   serializes phases by design.
4. `pkill -f <pattern>` inside an ssh remote command matches its own shell
   and kills the connection (exit 255). Quote patterns or use pgrep first.

## Where the analysis artifacts live

Reference nsys reps, the controlled optimization matrix (E0–E3), bs sweeps,
and the MFU/SM-utilization analysis scripts are kept internally — ask the
authors for a copy if you need same-caliber baselines to diff against.
