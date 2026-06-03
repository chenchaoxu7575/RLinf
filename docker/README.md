## Building Docker Images

RLinf provides a unified Dockerfile for both the math reasoning and embodied images, and can switch between the two images using the `BUILD_TARGET` build argument, which can be `reason` or `embodied-<env>`, where env is the name of the embodied environment (e.g., `maniskill_libero`, `behavior`, `metaworld`, `calvin`).
To build the Docker image, run the following command **in the RLinf root directory**:

```shell
export BUILD_TARGET=reason # or embodied for the embodied image
docker build -f docker/Dockerfile --build-arg BUILD_TARGET=$BUILD_TARGET -t rlinf:$BUILD_TARGET .
```

# Using the Docker Image

The built Docker image contains one or multiple Python virtual environments (venv) in the `/opt/venv` directory, depending on the `BUILD_TARGET`.

Currently, the reasoning image contains one venv named `reason` in `/opt/venv/reason`, while the embodied image contains three venvs named `openvla`, `openvla-oft` and `openpi` in `/opt/venv/`.

To switch to the desired venv, we have a built-in script `switch_env` that can switch among venvs in a single command.

```shell
source switch_env <env_name> # e.g., source switch_env openvla-oft, source switch_env openpi, etc.
```

## DreamZero VideoLoader Profiling

`Dockerfile.dreamzero_videoloader` builds a minimal CUDA 12.4 DreamZero
profiling image with Nsight Systems, NVTX, PyAV, torchcodec, and the DreamZero
repository under `/opt/dreamzero`. It is separate from the unified multi-target
Dockerfile so legacy Docker builders do not pull unrelated platform base images.

```shell
docker build -f docker/Dockerfile.dreamzero_videoloader \
  -t rlinf:embodied-dreamzero-videoloader .

bash scripts/run_dreamzero_videoloader_profile.sh \
  --dataset-root /tmp/rlinf_pub/datasets/DreamZero-DROID-Data-subset \
  --output-dir /tmp/rlinf_pub/results/dreamzero_videoloader_profile \
  --docker-gpus device=1
```

The runner can also download the default DROID `chunk-000` subset before
profiling. It writes summary CSV metrics, per-step CSV timings, and per-backend
Nsight `.nsys-rep` files. Torch profiler Chrome traces are intentionally run in
a separate optional pass with `--emit-torch-trace`, because combining torch
profiler and Nsight Systems can create CUPTI subscriber conflicts. By default,
the runner stores datasets, outputs, and Hugging Face cache under
`/tmp/rlinf_pub` to avoid consuming home directory quota. Set
`RLINF_PROFILE_BASE` or pass explicit paths to change this.

In Nsight Systems, filter NVTX ranges by `dreamzero.profile.step`,
`dreamzero.profile.next_batch`, `dreamzero.decode_video.pyav`,
`dreamzero.decode_video.torchcodec`, `dreamzero.dataset.transform`, or
`dreamzero.collate` to inspect the dataloader timeline.
