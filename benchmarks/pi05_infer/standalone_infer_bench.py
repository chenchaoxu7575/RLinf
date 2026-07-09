# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone pi0.5 bs=1 inference latency benchmark (no Ray, no env stack).

Reproduces the rollout worker's production inference path
(``OpenPi0ForRLActionPrediction.predict_action_batch`` with
``torch.compile max-autotune``, #968) on dummy observations shaped exactly
like the dummy Turtle2 realworld env (3x 128x128x3 uint8 cameras + 7-dim
tcp_pose state + text prompt), so a kernel engineer can measure and profile
without the RLinf scheduler or a robot.

Baseline (RTX PRO 5000 Blackwell sm_120, bs=1, num_steps=10, compiled):
e2e predict ~58.9 ms wall clock. See REPRO_GUIDE.md next to this file for
the full phase breakdown, nsys instructions, and known pitfalls.

Usage (from the repo root, inside the benchmark container):
    python benchmarks/pi05_infer/standalone_infer_bench.py \
        --model-path /workspace/rlinf_pub/models/RLinf-Pi05-LIBERO-SFT

    # phase breakdown (sync-timed decomposition of predict_action_batch)
    python benchmarks/pi05_infer/standalone_infer_bench.py --phases ...

    # under nsys (see REPRO_GUIDE.md for the full command)
    nsys profile ... --capture-range=cudaProfilerApi \
        python benchmarks/pi05_infer/standalone_infer_bench.py --cuda-profiler ...
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

# Allow running from a source checkout without installing rlinf.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/workspace/rlinf_pub/models/RLinf-Pi05-LIBERO-SFT",
        help="Checkpoint dir (safetensors + <asset_id>/norm_stats.json).",
    )
    parser.add_argument(
        "--config-name",
        default="pi05_turtle",
        help="openpi TrainConfig name (pi05_turtle = action_horizon 50).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-steps", type=int, default=10, help="Flow-matching denoise steps."
    )
    parser.add_argument(
        "--action-chunk", type=int, default=50, help="Executed action chunk length."
    )
    parser.add_argument(
        "--action-dim", type=int, default=6, help="Environment action dim."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=3,
        help="Camera views fed to the model (1 main + N-1 extra views).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Input camera resolution; the openpi transform resizes to 224.",
    )
    parser.add_argument(
        "--state-dim", type=int, default=7, help="Env state dim (turtle: xyz + quat)."
    )
    parser.add_argument(
        "--prompt",
        default="Press the button with the end-effector.",
        help="Task description (tokenized/padded to 200 tokens by the transform).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile (eager baseline; production runs compiled).",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        help="torch.compile mode for the production path.",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--phases",
        action="store_true",
        help="Also print a sync-timed per-phase decomposition of predict.",
    )
    parser.add_argument(
        "--cuda-profiler",
        action="store_true",
        help="Wrap the timed loop in torch.cuda.profiler.start()/stop() so that "
        "nsys --capture-range=cudaProfilerApi records only steady-state iters.",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Construct the model the same way the rollout worker does.

    Mirrors ``rollout.model`` in
    evaluations/realworld/realworld_dummy_turtle2_eval_pi05_nsys_serial.yaml,
    but calls the openpi builder directly instead of going through
    ``rlinf.models.get_model`` (which needs a Ray Worker for device placement).
    """
    from rlinf.models.embodiment.openpi import get_model

    cfg = OmegaConf.create(
        {
            "model_path": args.model_path,
            "precision": None,  # openpi manages its own per-param dtypes
            "openpi": {
                "config_name": args.config_name,
                "num_images_in_input": args.num_images,
                "noise_level": 0.5,
                "action_chunk": args.action_chunk,
                "num_steps": args.num_steps,
                "train_expert_only": True,
                "action_env_dim": args.action_dim,
                "noise_method": "flow_sde",
                "add_value_head": False,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": True,
            },
            "openpi_data": None,
        }
    )
    model = get_model(cfg).to(args.device).eval()
    if not args.no_compile:
        model.enable_torch_compile(mode=args.compile_mode)
    return model


def make_env_obs(args: argparse.Namespace) -> dict:
    """Dummy observation batch in the canonical env-output format.

    Matches ``EmbodiedIOStruct.prepare_observations`` (rlinf/data/
    embodied_io_struct.py) fed by the dummy Turtle2 env: CPU tensors, uint8
    HWC images, one main camera plus (num_images - 1) extra views stacked on
    dim 1. H2D transfer happens inside predict, as in production.
    """
    g = torch.Generator().manual_seed(args.seed)
    hw = (args.image_size, args.image_size)

    def _img(*lead: int) -> torch.Tensor:
        return torch.randint(0, 256, (*lead, *hw, 3), generator=g, dtype=torch.uint8)

    num_extra = args.num_images - 1
    return {
        "main_images": _img(args.batch_size),
        "wrist_images": None,
        "extra_view_images": _img(args.batch_size, num_extra) if num_extra else None,
        "states": torch.randn(
            args.batch_size, args.state_dim, generator=g, dtype=torch.float32
        ),
        "task_descriptions": [args.prompt] * args.batch_size,
    }


def _stats_ms(samples: list[float]) -> str:
    qs = statistics.quantiles(samples, n=10)
    return (
        f"mean {statistics.mean(samples):7.2f}  p50 {statistics.median(samples):7.2f}  "
        f"p90 {qs[8]:7.2f}  min {min(samples):7.2f}  max {max(samples):7.2f}"
    )


def run_e2e(model, env_obs: dict, args: argparse.Namespace) -> None:
    """Time N full predict_action_batch calls: CPU wall clock + GPU span."""
    print(f"warmup x{args.warmup} (absorbs compile/autotune; may take minutes) ...")
    for _ in range(args.warmup):
        with torch.no_grad():
            model.predict_action_batch(env_obs, mode="eval")
    torch.cuda.synchronize()

    if args.cuda_profiler:
        torch.cuda.profiler.start()

    wall_ms, gpu_ms = [], []
    for i in range(args.iters):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with torch.cuda.nvtx.range(f"bench/iter{i}"):
            t0 = time.perf_counter()
            start_evt.record()
            with torch.no_grad():
                model.predict_action_batch(env_obs, mode="eval")
            end_evt.record()
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - t0) * 1e3)
        gpu_ms.append(start_evt.elapsed_time(end_evt))

    if args.cuda_profiler:
        torch.cuda.profiler.stop()

    print(f"\n=== e2e predict_action_batch (bs={args.batch_size}, n={args.iters}) ===")
    print(f"  cpu wall clock [ms]  {_stats_ms(wall_ms)}")
    # GPU span (first->last GPU work), comparable to nsys projection, not busy time.
    print(f"  gpu event span [ms]  {_stats_ms(gpu_ms)}")


def run_phases(model, env_obs: dict, args: argparse.Namespace) -> None:
    """Sync-timed decomposition of one predict call into its NVTX phases.

    The GPU phases are re-invoked with the exact tensors captured from a
    real predict call (via a temporary wrapper around _build_prefix_cache).
    Re-deriving the intermediates independently can yield tensors with a
    different memory format (NCHW-contiguous instead of channels_last),
    which makes torch.compile recompile a slower vision-tower variant and
    overstate the prefix phase. The sum can still differ slightly from the
    e2e number (thread-pool overlap, allocator effects).
    """
    from openpi.models import model as _model

    def timed(name: str, fn):
        # Untimed warmup call: invoking the internals directly (instead of via
        # predict_action_batch) can trigger one dynamo recompile on the first
        # call, which must not land in the timed window.
        out = fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = fn()
        torch.cuda.synchronize()
        acc[name] = (time.perf_counter() - t0) / args.iters * 1e3
        return out

    acc: dict[str, float] = {}

    # Capture the prefix-phase inputs/outputs from one genuine predict so the
    # timed GPU phases see bit-identical tensors (shape, dtype, and strides).
    captured: dict[str, tuple] = {}
    orig_build_prefix_cache = model._build_prefix_cache

    def _capturing_build_prefix_cache(images, img_masks, lang_tokens, lang_masks):
        out = orig_build_prefix_cache(images, img_masks, lang_tokens, lang_masks)
        captured["prefix_args"] = (images, img_masks, lang_tokens, lang_masks)
        captured["prefix_out"] = out
        return out

    model._build_prefix_cache = _capturing_build_prefix_cache
    try:
        with torch.no_grad():
            model.predict_action_batch(env_obs, mode="eval")
    finally:
        model._build_prefix_cache = orig_build_prefix_cache
    assert "prefix_args" in captured, (
        "predict_action_batch did not call _build_prefix_cache; "
        "the phase decomposition no longer matches the model code."
    )
    images, img_masks, lang_tokens, lang_masks = captured["prefix_args"]
    _, prefix_pad_masks, past_key_values = captured["prefix_out"]

    with torch.no_grad():
        # CPU-side phases (memory-format independent, re-derived per call).
        to_process = timed(
            "predict/obs_processor", lambda: model.obs_processor(env_obs)
        )
        transformed = timed(
            "predict/input_transform",
            lambda: model.input_transform(to_process, transpose=False),
        )
        observation = timed(
            "predict/precision+from_dict",
            lambda: _model.Observation.from_dict(
                model.precision_processor(dict(transformed))
            ),
        )
        preprocessed = timed(
            "denoise/preprocess",
            lambda: model._preprocess_observation(observation, train=False),
        )
        state = preprocessed[4]
        # GPU phases, replayed with the captured production tensors.
        timed(
            "prefix (embed+mask+vlm_forward)",
            lambda: orig_build_prefix_cache(images, img_masks, lang_tokens, lang_masks),
        )

        def denoise_loop():
            # Eval path replica: every step is flow_ode (denoise_inds == -1).
            x_t = model.sample_noise(
                (args.batch_size, model.config.action_horizon, model.config.action_dim),
                args.device,
            )
            for idx in range(args.num_steps):
                x_t_mean, x_t_std, value_t, _v_t = model.sample_mean_var_val(
                    x_t,
                    idx,
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    "flow_ode",
                    args.num_steps,
                    True,
                )
                x_t = x_t_mean + model.sample_noise(x_t.shape, args.device) * x_t_std
                model.get_logprob_norm(x_t, x_t_mean, x_t_std)
            return x_t

        actions = timed(f"denoise/loop (x{args.num_steps} steps)", denoise_loop)
        timed(
            "predict/output_transform",
            lambda: model.output_transform(
                {"actions": actions, "state": observation.state}
            ),
        )

    total = sum(acc.values())
    print(f"\n=== per-phase wall clock (bs={args.batch_size}, n={args.iters}) ===")
    for name, ms in acc.items():
        print(f"  {name:<34} {ms:8.2f} ms  ({100 * ms / total:4.1f}%)")
    print(f"  {'SUM':<34} {total:8.2f} ms")


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA device required for this benchmark."
    torch.cuda.set_device(args.device)

    print(f"gpu: {torch.cuda.get_device_name(args.device)}")
    print(f"torch: {torch.__version__}")
    compile_desc = "eager" if args.no_compile else args.compile_mode
    print(
        f"config: {args.config_name} bs={args.batch_size} num_steps={args.num_steps} "
        f"chunk={args.action_chunk} images={args.num_images}x{args.image_size} "
        f"compile={compile_desc}"
    )

    model = build_model(args)
    env_obs = make_env_obs(args)

    run_e2e(model, env_obs, args)
    if args.phases:
        run_phases(model, env_obs, args)


if __name__ == "__main__":
    main()
