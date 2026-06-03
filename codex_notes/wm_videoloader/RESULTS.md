# DreamZero WorldModel VideoLoader Profile Results

## Scope

This note records the profiling run for the optimized DreamZero WorldModel SFT path using the `torchcodec` video backend. Raw Nsight reports and sqlite exports are kept locally under `codex_notes/wm_videoloader/results/` and are intentionally not committed.

## Run Configuration

| Field | Value |
| --- | --- |
| Run directory | `codex_notes/wm_videoloader/results/worldmodel_videoloader_cpu_nsys/20260602_223623` |
| Script | `scripts/run_worldmodel_videoloader_nsys.sh --rank-profile --profile-ranks 0,1 --max-steps 12` |
| Model config | `examples/sft/config/droid_sft_dreamzero_14b.yaml` |
| Video backend | `torchcodec` |
| GPUs | 8 x H20 |
| Global batch size | 8 |
| Micro batch size | 1 per GPU |
| Gradient accumulation | 1 |
| Input video shape per rank | `[1, 3, 33, 352, 640]` |
| DataLoader workers per rank | 4 |
| DataLoader prefetch factor | 8 |
| Profiled ranks | ActorGroup rank 0 and rank 1 |
| Nsight capture mode | Worker-rank NVTX capture range: `sft.worker.profile_window` |

## Key Results

| Metric | Rank 0 | Rank 1 | Notes |
| --- | ---: | ---: | --- |
| Steady-state step time avg | 23.716 s | 23.716 s | Excludes first step |
| Steady-state step time min | 22.598 s | 22.599 s | From step CSV |
| Steady-state step time max | 24.376 s | 24.376 s | From step CSV |
| `next_batch_ms` avg | 0.500 ms | 0.669 ms | Main worker wait for DataLoader batch |
| `next_batch_ms` max | 0.671 ms | 1.109 ms | Main worker wait for DataLoader batch |
| Data wait fraction avg | 0.000021 | 0.000028 | `next_batch_ms / step_ms` |
| `dreamzero.dataset.__getitem__` avg | 1568.1 ms | 1497.5 ms | Nsight NVTX, DataLoader worker side |
| `dreamzero.dataset.transform` avg | 1317.4 ms | 1242.3 ms | Nsight NVTX, DataLoader worker side |
| Torchcodec decode count | 36 | 36 | 12 samples x 3 camera views |
| Torchcodec decode avg per view | 64.1 ms | 62.9 ms | `dreamzero.video.decode.lerobot_decode.torchcodec` |
| Torchcodec decode approx per sample | 192.3 ms | 188.7 ms | 3 views per sample |
| Nsight report size | 57.7 MB | 55.3 MB | `.nsys-rep` |
| Nsight sqlite size | 184.8 MB | 176.8 MB | Exported sqlite |

## GPU Utilization

| GPU | Peak memory MiB | Peak utilization |
| ---: | ---: | ---: |
| 0 | 97261 | 100% |
| 1 | 97339 | 100% |
| 2 | 97323 | 100% |
| 3 | 97323 | 100% |
| 4 | 97322 | 100% |
| 5 | 97322 | 100% |
| 6 | 97322 | 100% |
| 7 | 97362 | 100% |

## Interpretation

The optimized `torchcodec` DataLoader path is hidden by the current WorldModel training step. Although the DataLoader worker side still spends about 1.5 seconds per sample in dataset work, the main training worker only waits around 0.5 to 0.7 ms for `next(self.data_iter)` during steady-state steps. The current training workload is compute and memory dominated: per-GPU micro batch size is already 1, and peak GPU memory is about 97.3 GiB on 97.9 GiB H20 devices.

The first rank-profile attempt did not preserve Nsight reports because waiting for Ray worker teardown was not a reliable report-finalization point. The successful run uses an explicit worker NVTX capture window and `--capture-range-end=stop-shutdown --kill=none` so Nsight finalizes reports when the profiled step window ends.
