"""A/B micro-benchmark for the multi_channel_pg pinned-staging fix.

Measures the exact device<->host copies the fix changes on the GLOO-fallback
path (`_no_accel_ccl and device==ACCEL`), pageable (old) vs pinned (new):

  D2H (sender stage):   old = tensor.to("cpu")            new = _stage_to_pinned_cpu(tensor)
  H2D (receiver apply): old = accel.copy_(pageable_cpu)   new = accel.copy_(pinned_cpu)

The GLOO broadcast between them is byte-identical either way, so these copies
are the entire effect of the fix. Single process (D2H/H2D are local PCIe ops).
"""

import time

import torch

from rlinf.scheduler.collective.multi_channel_pg import MultiChannelProcessGroup

DTYPE = torch.bfloat16
SIZES_GB = [0.125, 1.0, 4.0]  # 128MB bucket, 1GB tensor, 4GB tensor
REPEATS = 5


def _gbps(nbytes: int, secs: float) -> float:
    return (nbytes / 1e9) / secs


def _time_d2h(gpu: torch.Tensor, pinned: bool) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if pinned:
        _ = MultiChannelProcessGroup._stage_to_pinned_cpu(gpu)
    else:
        _ = gpu.to("cpu")  # legacy pageable path
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _time_h2d(host: torch.Tensor, gpu_dst: torch.Tensor) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu_dst.copy_(host, non_blocking=True)  # _copy_to_accel_tensor
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    assert torch.cuda.is_available(), "benchmark requires a CUDA device"
    print(f"device={torch.cuda.get_device_name(0)} dtype={DTYPE}\n")
    print(f"{'size':>8} {'leg':>4} {'pageable GB/s':>14} {'pinned GB/s':>12} {'speedup':>8}")
    for gb in SIZES_GB:
        numel = int(gb * 1e9 / 2)  # bf16 = 2 bytes
        nbytes = numel * 2
        gpu = torch.empty(numel, dtype=DTYPE, device="cuda").normal_()
        pageable = gpu.to("cpu")
        pinned = MultiChannelProcessGroup._stage_to_pinned_cpu(gpu)
        gpu_dst = torch.empty_like(gpu)

        # warmup
        _time_d2h(gpu, pinned=False); _time_d2h(gpu, pinned=True)
        _time_h2d(pageable, gpu_dst); _time_h2d(pinned, gpu_dst)

        d2h_page = min(_time_d2h(gpu, pinned=False) for _ in range(REPEATS))
        d2h_pin = min(_time_d2h(gpu, pinned=True) for _ in range(REPEATS))
        h2d_page = min(_time_h2d(pageable, gpu_dst) for _ in range(REPEATS))
        h2d_pin = min(_time_h2d(pinned, gpu_dst) for _ in range(REPEATS))

        print(
            f"{gb:>6}GB {'D2H':>4} {_gbps(nbytes, d2h_page):>14.1f} "
            f"{_gbps(nbytes, d2h_pin):>12.1f} {_gbps(nbytes, d2h_pin) / _gbps(nbytes, d2h_page):>7.1f}x"
        )
        print(
            f"{gb:>6}GB {'H2D':>4} {_gbps(nbytes, h2d_page):>14.1f} "
            f"{_gbps(nbytes, h2d_pin):>12.1f} {_gbps(nbytes, h2d_pin) / _gbps(nbytes, h2d_page):>7.1f}x"
        )
        del gpu, pageable, pinned, gpu_dst
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
