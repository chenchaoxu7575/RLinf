# Pinned-memory weight sync over the GLOO fallback — experiment record

Archive of the weight-sync D2H/H2D pinned-memory optimization: the problem, the two
implementation approaches we prototyped, and the A/B measurements. The **shipped**
fix is the transport-layer version (separate PR); this branch preserves the
**syncer-layer** prototype and the full comparison as a record.

## Problem

When an accelerator collective can't use CCL and RLinf routes it over the **GLOO
fallback** (`MultiChannelProcessGroup._no_accel_ccl` — heterogeneous accelerator
models, CPU-only, or unsupported CCL; e.g. H20 trainer + L20 rollout), tensors must
move device → host → socket → host → device. Those host↔device copies used
**pageable** host memory, which copies at only a fraction of PCIe bandwidth. This
dominates cross-node weight sync (a full model per sync).

The fix: stage those copies through **pinned (page-locked)** host memory. The two
host↔device legs are inherent to the GLOO fallback (GLOO can't touch device memory);
we only change the staging buffer from pageable to pinned — same copies, same
byte count, ~5–25× the bandwidth.

## Two approaches prototyped

### A) Syncer layer — `PatchWeightSyncer` (this branch)
Stage the patch in `sync()` (D2H) / `apply()` (H2D) through pinned host memory.
- Pros: localized to weight sync; easy to A/B in-code.
- Cons: **pushes transport-level host-staging into the sync algorithm (breaks the
  layering)**, and only speeds up weight sync — not other GLOO-fallback collectives.
- `patch_syncer.py` in this branch is the clean implementation, split across two
  commits: (1) the delta-patch pinned staging; (2) an optional full-weight
  bucket-streamed path (`RLINF_WEIGHT_SYNC_FORCE_FULL`) that stages through a
  **single reusable pinned "arena" buffer preallocated at init, sized to the largest
  bucket** — so host-pinned memory stays bounded by one bucket, not the whole model.
- `patch_syncer_measurement_reference.py` (in this folder) is the *as-run measurement*
  build that produced the 2-node numbers: same idea but with per-param pinned buffers
  (heavier), plus in-code D2H/H2D CUDA timing, path diagnostics, and NVTX ranges.

### B) Transport layer — `MultiChannelProcessGroup` (shipped)
Pin the host staging buffers where the GLOO fallback already does the device↔host
copies (`send`/`broadcast` D2H via `_stage_to_pinned_cpu`; `recv`/broadcast host
buffers via `pin_memory=True`; the H2D in `_copy_to_accel_tensor` is already
`non_blocking`, so a pinned source is fast automatically).
- Pros: correct layer, one localized change, benefits **every** GLOO-fallback
  collective (weight sync, bucket sync, any GPU-tensor comm); no syncer changes.
- Buffers are allocated fresh per call; PyTorch's caching pinned-host allocator
  recycles them (a pool hit in steady state), which also handles async/multi-channel
  reuse safety.

We shipped **B**.

## A/B measurements (all H20, pageable → pinned)

| Setup | Payload | D2H | H2D |
|-------|---------|-----|-----|
| Isolated nsys (real `patch_syncer`, single process) | 805 MB delta | ~**24×** | ~**6×** |
| **2-node real GLOO**, syncer-layer, in-code timing | 8.53 GB full dense | 1.4–1.5 → ~50 GB/s (~**32×**) | 4.1–5.1 → ~26 GB/s (~**6×**) |
| Transport micro-benchmark (`bench_transport_pinned.py`) | 128 MB / 1 GB / 4 GB | 2 → 55 GB/s (~**24–29×**) | 11 → 55 GB/s (~**5×**) |

Three independent setups agree: pinned lifts D2H and H2D to ~55 GB/s ≈ the H20 PCIe
line rate. Pageable D2H is especially bad (~2 GB/s); pageable H2D is ~11 GB/s.

Note the two legs are **local PCIe** copies and NIC-independent — the cross-node
network (which rode the management NIC in our 2-node run) affects total wall-clock
but not these per-copy bandwidths.

## Files in this archive

- `rlinf/hybrid_engines/weight_syncer/patch_syncer.py` — syncer-layer clean
  implementation (delta pinned staging + full-weight init-preallocated bucket arena).
- `docs/dev_notes/pinned_weight_sync/patch_syncer_measurement_reference.py` — the
  as-run measurement build (per-param pinned buffers + timing + diagnostics + NVTX).
- `examples/embodiment/config/realworld_dummy_turtle2_ppo_pi05_2node_{pageable_baseline,pinnedtest}.yaml`
  — the 2-node A/B configs (`transport_device: cpu`; baseline sets
  `RLINF_DISABLE_PINNED_WEIGHT_SYNC=1`). Depend on `pi05_turtle` (also included).
- `docs/dev_notes/pinned_weight_sync/bench_transport_pinned.py` — the transport-layer
  micro-benchmark that produced the third row above.

## Practical notes learned

- With `transport_device=cpu`, the syncer pre-moves to host and the transport does
  pure CPU↔CPU; the transport-layer fix is exercised on the `transport_device=accel`
  (GPU tensors into the collective) path instead.
- Profiling gotchas from the 2-node run: (1) profiling is configured under
  `cluster.profiling` (a `NsightConfig`), not a top-level `nsight_profiler:` block —
  the latter is silently ignored; (2) `cuda-memory-usage=true` can stall nsys on a
  full-model transfer; (3) `capture-range-end: repeat-shutdown:N` must match the
  number of profiled cudaProfiler ranges or the report never flushes.
