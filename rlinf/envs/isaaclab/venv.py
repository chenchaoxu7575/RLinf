# Copyright 2025 The RLinf Authors.
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

import os
import shlex
import signal
import subprocess
import sys
from multiprocessing.connection import Connection

import torch
import torch.multiprocessing as mp

from rlinf.utils import nsight_profiler

from .utils import CloudpickleWrapper


def _maybe_wrap_isaac_spawn_with_nsys() -> None:
    """Give the spawned Isaac Sim subprocess its OWN ``nsys profile`` session.

    Isaac Sim runs in a ``torch.multiprocessing`` spawn child that a fork-following
    nsys attached to the parent EnvWorker fails to capture CUDA for (the parent's
    ``capture-range=cudaProfilerApi`` window never propagates to the child). Instead
    of relying on fork-follow, wrap the child interpreter itself: point
    ``multiprocessing``'s spawn executable at a tiny shell that re-execs the real
    interpreter under ``nsys profile``. The child already drives
    ``torch.cuda.profiler.start()/stop()`` (see the ``profile_start``/``profile_stop``
    commands), so in its own single-process session ``capture-range=cudaProfilerApi``
    gates cleanly to the requested RL steps -- no warmup, correct sim CUDA.

    Gated by ``_RLINF_ISAAC_NSYS_ON=1`` (set per-rank by ``EnvWorker.init_worker``).
    The parent EnvWorker must NOT itself be nsys-wrapped (drop EnvGroup from
    ``cluster.nsight.worker_groups``) or the child's nsys would nest and error.
    """
    if os.environ.get("_RLINF_ISAAC_NSYS_ON") != "1":
        return
    out_dir = os.environ.get("RLINF_ISAAC_NSYS_OUT", "/tmp")
    os.makedirs(out_dir, exist_ok=True)
    real_py = sys.executable
    trace = os.environ.get("RLINF_ISAAC_NSYS_TRACE", "cuda,cudnn,cublas,nvtx,vulkan")
    # capture-range: "cudaProfilerApi" step-gates via the child's own
    # torch.cuda.profiler.start()/stop(); "none" captures the whole child (slice
    # by timestamp in analysis). Set via RLINF_ISAAC_NSYS_CAPTURE.
    capture = os.environ.get("RLINF_ISAAC_NSYS_CAPTURE", "cudaProfilerApi")
    cap_flags = ""
    if capture == "nvtx":
        # Step-gate on an NVTX range the child pushes on profile_start/stop
        # (cudaProfilerApi capture-range is NOT honored for a spawn child, but
        # nsys DOES watch NVTX ranges to open/close the capture window).
        # stop-shutdown: write the report and end the session the moment the first
        # ISAAC_PROFILE_WINDOW closes (mid-run), so finalization does NOT depend on
        # a clean shutdown or -d (both of which failed to write a nvtx-gated report).
        cap_flags = (
            " --capture-range=nvtx"
            " --nvtx-capture=ISAAC_PROFILE_WINDOW"
            " --capture-range-end=stop-shutdown"
        )
    elif capture and capture != "none":
        cap_flags = f" --capture-range={capture} --capture-range-end=stop"
    # The runner never sends "close" and tears the daemon child down at run end,
    # so a large capture=none trace never gets finalized to -o. With a duration
    # (seconds), nsys stops collection and WRITES the report mid-run while the
    # sim keeps running (--kill=none), sidestepping the shutdown-finalize gap.
    duration = os.environ.get("RLINF_ISAAC_NSYS_DURATION", "").strip()
    dur_flags = f" -d {int(duration)} --kill=none" if duration else ""
    rank = os.environ.get("_RLINF_ISAAC_NSYS_RANK", "NA")
    prefix = os.path.join(out_dir, f"rlinf_nsight_IsaacSim_rank{rank}_%p")
    wrapper = os.path.join(out_dir, "isaac_nsys_wrapper.sh")
    # Common prologue: scrub every nsys injection var inherited from a (possibly)
    # profiled parent so the child's own nsys starts clean and does not nest/error.
    prologue = (
        "#!/bin/bash\n"
        "unset LD_PRELOAD CUDA_INJECTION64_PATH\n"
        "for v in $(env | sed -n 's/^\\(NSYS[A-Z0-9_]*\\|QUADD[A-Z0-9_]*\\)=.*/\\1/p'); do unset \"$v\"; done\n"
    )
    if capture == "session":
        # nsys launch: instrument the child but collect nothing until the child
        # drives `nsys start`/`nsys stop` per RL step (see _isaac_nsys_session_*),
        # yielding ONE report file per RL step. Unique session per child via $$.
        script = (
            prologue
            + 'export RLINF_ISAAC_SESSION="isaac_$$"\n'
            + "exec nsys launch"
            + f" -t {shlex.quote(trace)}"
            + " --cuda-graph-trace=node"
            + ' --session-new="$RLINF_ISAAC_SESSION"'
            + f" {shlex.quote(real_py)} \"$@\"\n"
        )
    else:
        # nsys profile: single report. No --trace-fork-before-exec (following
        # Isaac's forked Kit render subprocesses corrupted event processing and
        # blocked report finalization under capture-range=nvtx + -d).
        script = (
            prologue
            + "exec nsys profile"
            + f" -t {shlex.quote(trace)}"
            + f"{cap_flags}"
            + f"{dur_flags}"
            + " --cuda-graph-trace=node"
            + f" -o {shlex.quote(prefix)}"
            + " --force-overwrite=true"
            + f" {shlex.quote(real_py)} \"$@\"\n"
        )
    with open(wrapper, "w") as f:
        f.write(script)
    os.chmod(wrapper, 0o755)
    mp.set_executable(wrapper)


def _isaac_nsys_session_start(step_idx) -> None:
    """Start a fresh nsys collection for one RL step (session mode).

    Writes to a per-RL-step output file so each rollout window becomes its own
    report. No-op unless ``RLINF_ISAAC_NSYS_CAPTURE=session``.
    """
    # Only ranks that were actually nsys-launched (per-rank _RLINF_ISAAC_NSYS_ON)
    # have a session; others would call nsys start with session=None -> TypeError.
    if (
        os.environ.get("RLINF_ISAAC_NSYS_CAPTURE") != "session"
        or os.environ.get("_RLINF_ISAAC_NSYS_ON") != "1"
    ):
        return
    session = os.environ.get("RLINF_ISAAC_SESSION")
    out_dir = os.environ.get("RLINF_ISAAC_NSYS_OUT", "/tmp")
    # Tag the report with the originating env-worker rank (set by
    # EnvWorker.init_worker, inherited via os.environ) so multi-rank runs don't
    # collide and each file says which rank it came from.
    rank = os.environ.get("_RLINF_ISAAC_NSYS_RANK", "NA")
    out = os.path.join(
        out_dir, f"rlinf_nsight_IsaacSim_rank{rank}_step{step_idx}_{os.getpid()}"
    )
    r = subprocess.run(
        # NOTE: `nsys start` does NOT accept -t/--trace (that is set once on
        # `nsys launch`). NVTX/Vulkan get collected as long as launch's -t lists
        # them AND there are events in-window (sim/step needs _profiling_active).
        # The FIRST start of a session is slow to engage collection at 8-GPU
        # scale (was hitting a 120s TypeError timeout), so allow generous time;
        # profile steps [1,2,3] and discard step 1 to absorb this cold start.
        ["nsys", "start", "--session", session, "-c", "none", "-o", out, "-f", "true"],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r.returncode != 0:
        print(
            f"[isaac-nsys-child] nsys start rc={r.returncode} "
            f"out={r.stdout[-200:]!r} err={r.stderr[-200:]!r}",
            flush=True,
        )


def _isaac_nsys_session_stop() -> None:
    """Stop the current nsys collection and write its per-RL-step report."""
    if (
        os.environ.get("RLINF_ISAAC_NSYS_CAPTURE") != "session"
        or os.environ.get("_RLINF_ISAAC_NSYS_ON") != "1"
    ):
        return
    session = os.environ.get("RLINF_ISAAC_SESSION")
    r = subprocess.run(
        ["nsys", "stop", "--session", session],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r.returncode != 0:
        print(
            f"[isaac-nsys-child] nsys stop rc={r.returncode} "
            f"out={r.stdout[-200:]!r} err={r.stderr[-200:]!r}",
            flush=True,
        )


def _torch_worker(
    child_remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper,
    action_queue: mp.Queue,
    obs_queue: mp.Queue,
    reset_idx_queue: mp.Queue,
):
    parent_remote.close()
    # When this child runs under its own nsys, the runner tears the daemon child
    # down (SIGTERM) at the end of the run WITHOUT sending "close", so nsys never
    # gets to flush its report. Turn SIGTERM into a KeyboardInterrupt so the loop
    # below exits cleanly (closing the sim) and the wrapping nsys finalizes.
    if os.environ.get("_RLINF_ISAAC_NSYS_ON") == "1":

        def _finalize_on_sigterm(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _finalize_on_sigterm)
    # Route Omniverse Kit's OWN internal profiler zones (render / physics / fabric
    # / USD update, thousands of ranges) to NVTX so nsys captures them. Kit
    # defaults to the Tracy backend and emits NO NVTX; select the nvtx carb
    # profiler backend + profile-from-start via kit args on sys.argv, which
    # AppLauncher/SimulationApp forwards to carb. Gated by RLINF_ISAAC_KIT_NVTX=1.
    if os.environ.get("RLINF_ISAAC_KIT_NVTX") == "1":
        sys.argv += [
            "--/app/profilerBackend=nvtx",
            "--/app/profileFromStart=true",
        ]
    env_fn = env_fn_wrapper.x
    isaac_env, sim_app = env_fn()
    device = isaac_env.device
    _profile_win = None  # NVTX range id for the ISAAC_PROFILE_WINDOW capture gate
    try:
        while True:
            try:
                cmd = child_remote.recv()
            except EOFError:
                child_remote.close()
                break
            profile_step = None
            if isinstance(cmd, tuple):
                cmd, profile_step = cmd
            if cmd == "reset":
                reset_index, reset_seed = reset_idx_queue.get()
                with nsight_profiler.profile_range("sim/reset"):
                    if reset_index is None:
                        reset_result = isaac_env.reset(seed=reset_seed)
                    else:
                        reset_result = isaac_env.reset(
                            seed=reset_seed, env_ids=reset_index.to(device)
                        )
                obs_queue.put(reset_result)
            elif cmd == "step":
                input_action = action_queue.get()
                with nsight_profiler.profile_range("sim/step"):
                    step_result = isaac_env.step(input_action)
                obs_queue.put(step_result)
            # Per-RL-step nsys profiling. In-process capture-range triggers
            # (cudaProfilerApi, NVTX) are NOT honored for this spawn child, so in
            # "session" mode the child is launched under `nsys launch` and drives
            # collection externally via `nsys start`/`nsys stop` -- one report file
            # per RL step (each rollout window, containing all its env steps).
            elif cmd == "profile_start":
                try:
                    # 1) begin nsys collection for this RL step, THEN
                    # 2) enable _profiling_active so nsight_profiler.profile_range
                    #    ("sim/step"/"sim/reset") actually emits its NVTX ranges
                    #    inside the window, THEN 3) push an in-window RL-step marker.
                    _isaac_nsys_session_start(profile_step)
                    nsight_profiler.start_profile(drive_cuda_profiler=False)
                    torch.cuda.nvtx.range_push(f"RL_step_{profile_step}")
                    _profile_win = True
                    print(
                        f"[isaac-nsys-child] profile_start: RL step {profile_step} "
                        "nsys start + NVTX active",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[isaac-nsys-child] profile_start FAILED: {e!r}", flush=True
                    )
            elif cmd == "profile_stop":
                try:
                    if _profile_win:
                        torch.cuda.nvtx.range_pop()
                        _profile_win = False
                    nsight_profiler.stop_profile(drive_cuda_profiler=False)
                    _isaac_nsys_session_stop()  # write this RL step's report last
                    print(
                        "[isaac-nsys-child] profile_stop: NVTX inactive + nsys stop",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[isaac-nsys-child] profile_stop FAILED: {e!r}", flush=True)
            elif cmd == "close":
                isaac_env.close()
                child_remote.close()
                sim_app.close()
                break
            elif cmd == "device":
                child_remote.send(isaac_env.device)
            else:
                child_remote.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        child_remote.close()
    finally:
        try:
            isaac_env.close()
        except Exception as e:
            print(f"IsaacLab Env Closed with error: {e}")


class SubProcIsaacLabEnv:
    def __init__(self, env_fn):
        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe(duplex=True)
        self.action_queue = ctx.Queue()
        self.obs_queue = ctx.Queue()
        self.reset_idx = ctx.Queue()
        args = (
            self.child_remote,
            self.parent_remote,
            CloudpickleWrapper(env_fn),
            self.action_queue,
            self.obs_queue,
            self.reset_idx,
        )
        # Set the nsys wrapper AFTER Pipe/Queue creation so the multiprocessing
        # resource_tracker (spawned lazily on the first semaphore) uses the plain
        # interpreter; only the Isaac Sim process below launches under nsys.
        _maybe_wrap_isaac_spawn_with_nsys()
        self.isaac_lab_process = ctx.Process(
            target=_torch_worker, args=args, daemon=True
        )
        self.isaac_lab_process.start()
        self.child_remote.close()

    def reset(self, seed=None, env_ids=None):
        self.parent_remote.send("reset")
        self.reset_idx.put((env_ids, seed))
        obs, info = self.obs_queue.get()
        return obs, info

    def step(self, action: torch.Tensor):
        """
        action : (bs, action_dim)
        """
        self.parent_remote.send("step")
        self.action_queue.put(action)
        env_step_result = self.obs_queue.get()
        return env_step_result

    def close(self):
        self.parent_remote.send("close")
        self.isaac_lab_process.join()
        self.isaac_lab_process.terminate()

    def device(self):
        self.parent_remote.send("device")
        return self.parent_remote.recv()

    def start_profile(self, step_idx=None):
        """Open the nsys capture window inside the Isaac Sim subprocess.

        The RL step index is forwarded so the child can name a per-RL-step nsys
        report (one file per RL step / rollout, each containing all its env steps).
        """
        self.parent_remote.send(("profile_start", step_idx))

    def stop_profile(self):
        """Close the nsys capture window inside the Isaac Sim subprocess."""
        self.parent_remote.send(("profile_stop", None))
