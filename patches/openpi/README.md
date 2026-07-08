# openpi site-packages patches (profiling + bs=1 latency)

Patched copies of three openpi files that must live **inside the openpi
install** (they change code that runs before/inside openpi's own import and
forward paths, so they cannot be applied from RLinf's side). Apply with
`./apply.sh`; directory layout mirrors `site-packages/openpi/`.

Based on the openpi build shipped in the `rlinf0.2_0409` (H20) and
`rlinf-blackwell` (RTX PRO 5000, sm_120) containers, 2026-07. Before applying
to a different openpi version, diff against the target first — every change is
additive and clearly marked, so rebasing is mechanical.

All changes are unconditional (no env-var gates).

| file | change |
|---|---|
| `models_pytorch/pi0_pytorch.py` | 1) fine-grained NVTX in `embed_prefix` (`prefix/vision_siglip`, `prefix/lang_embed`) 2) batched SigLIP: embed all camera views in one ViT call (batch = num_views x B) instead of a per-view loop. At bs=1 this cut `prefix/vision_siglip` 9.5 -> 6.4 ms/predict (RTX PRO 5000, compiled); verified equal to the loop path (bit-exact on a stubbed ViT) |
| `models/model.py` | NVTX split inside `Observation.from_dict` (`from_dict/img_norm`, `from_dict/pack_typecheck`) — attributes the typecheck CPU cost; no behavior change |
| `shared/array_typing.py` | `@at.typecheck` decorator no-op'd at its definition site: jaxtyped+beartype validation costs ~3.2 ms/predict of pure CPU on the torch rollout path (`from_dict` 3.5 -> 0.3 ms). Note: a guard in RLinf's `openpi_action_model.py` (476e43d4) is NOT sufficient in the full worker — dataconfig imports `openpi.models.model` first, so classes are already decorated by the time it runs; only the definition site is import-order-proof |

Measured combined effect at bs=1, num_steps=10, torch.compile max-autotune,
RTX PRO 5000: e2e predict 65.8 -> 58.9 ms (see
`claude_mem/pi05_rollout_forward/nsys_sm120/README.md` for the controlled
matrix; those runs used env-gated versions of these same changes).
