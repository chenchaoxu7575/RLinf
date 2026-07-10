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

"""Per-step Nsight Systems profiling driver.

Pairs with the ``cluster.nsight`` launch wrapper: when ``cluster.nsight.steps``
is set, ``__post_init__`` auto-attaches ``capture-range=cudaProfilerApi`` to
the nsys command line, and this module's :func:`start_profile` /
:func:`stop_profile` drive ``torch.cuda.profiler.start()`` / ``stop()`` so
nsys only writes data inside the configured step windows.

Workers don't need any local state -- the runner toggles a module-level flag
and decorated methods consult it via :meth:`NsightProfiler.annotate`.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
from typing import Any, Callable, Optional

try:
    import nvtx as _nvtx  # type: ignore

    _NVTX_AVAILABLE = True
except ImportError:
    _nvtx = None  # type: ignore
    _NVTX_AVAILABLE = False

import torch

logger = logging.getLogger(__name__)

# Emit NVTX ranges via the optional ``nvtx`` Python package when present, else
# fall back to ``torch.cuda.nvtx`` (always available with a CUDA build). This
# keeps NVTX annotations working on minimal installs (e.g. the trocar image,
# which does not ship the ``nvtx`` wheel) without adding a hard dependency.
_TORCH_NVTX_AVAILABLE = hasattr(torch.cuda, "nvtx")


def _range_push(label: str, color: Optional[str], domain: Optional[str]):
    """Open an NVTX range; returns an opaque token for :func:`_range_pop`."""
    if _NVTX_AVAILABLE:
        return _nvtx.start_range(message=label, color=color, domain=domain)
    if _TORCH_NVTX_AVAILABLE:
        torch.cuda.nvtx.range_push(label)
        return "__torch__"
    return None


def _range_pop(token) -> None:
    """Close the range opened by :func:`_range_push`."""
    if token is None:
        return
    if token == "__torch__":
        torch.cuda.nvtx.range_pop()
    else:
        _nvtx.end_range(token)


# Module-level flag toggled by the runner around gated steps. Worker
# decorators read this directly (no per-worker profiler instance needed).
_profiling_active: bool = False


def is_profiling_active() -> bool:
    """Return whether step-gated profiling is currently in a capture window."""
    return _profiling_active


def start_profile(
    step_idx: Optional[int] = None, drive_cuda_profiler: bool = True
) -> None:
    """Open a profiling window for the current step.

    Calls ``torch.cuda.profiler.start()`` so nsys (running with
    ``capture-range=cudaProfilerApi``) begins writing data. The runner is
    expected to invoke this only when ``NsightConfig.should_profile_step``
    returns True for the current step.

    ``drive_cuda_profiler=False`` toggles only the NVTX-active flag without
    issuing ``cudaProfilerStart``. This is for a process that emits NVTX but
    does no CUDA work of its own (e.g. the parent ``EnvWorker``, which forwards
    all GPU work to an Isaac Sim subprocess): under
    ``capture-range=cudaProfilerApi`` only the *first* cudaProfilerStart/Stop
    pair in the process tree is honored, so a no-CUDA parent calling it would
    steal that slot and leave the real CUDA process (the child) uncaptured.
    """
    global _profiling_active
    if _profiling_active:
        return
    _profiling_active = True
    if drive_cuda_profiler:
        torch.cuda.profiler.start()
    if step_idx is not None:
        logger.info("Nsight profiler window opened at step %d", step_idx)


def stop_profile(drive_cuda_profiler: bool = True) -> None:
    """Close the current profiling window.

    See :func:`start_profile` for ``drive_cuda_profiler``.
    """
    global _profiling_active
    if not _profiling_active:
        return
    if drive_cuda_profiler:
        torch.cuda.profiler.stop()
    _profiling_active = False


@contextlib.contextmanager
def profile_range(
    message: str,
    color: Optional[str] = None,
    domain: Optional[str] = None,
):
    """Emit an NVTX range for a code block while profiling is active.

    The ``@NsightProfiler.annotate`` decorator equivalent for inline blocks
    (e.g. the Isaac Sim step inside the env subprocess loop, where a decorator
    cannot be applied). Emits nothing when :func:`is_profiling_active` is False.
    """
    if not _profiling_active:
        yield
        return
    token = _range_push(message, color, domain)
    try:
        yield
    finally:
        _range_pop(token)


class NsightProfiler:
    """Namespace for the NVTX annotation decorator used by worker methods.

    There is intentionally no per-worker instance: profiling-active state
    lives in a module-level flag toggled by the runner around the configured
    steps, and the launch-time decision of which worker groups are wrapped
    under ``nsys profile`` is owned by ``cluster.nsight.worker_groups`` (see
    ``rlinf/scheduler/cluster/config.py``). Workers don't need to know
    either.
    """

    @staticmethod
    def annotate(
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Callable:
        """Decorate a worker method to emit an NVTX range during profiling.

        The range is only emitted when the runner has currently opened a
        profiling window (i.e. :func:`is_profiling_active` returns True) and
        the ``nvtx`` Python package is importable. Otherwise the original
        method runs unchanged with no measurable overhead beyond a single
        boolean read.

        Example::

            class MyWorker(Worker):
                @NsightProfiler.annotate("rollout.predict")
                def predict(self, obs): ...
        """

        def decorator(func: Callable) -> Callable:
            label_default = message
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    if not _profiling_active:
                        return await func(*args, **kwargs)
                    label = label_default or func.__name__
                    token = _range_push(label, color, domain)
                    try:
                        return await func(*args, **kwargs)
                    finally:
                        _range_pop(token)

                return async_wrapper

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _profiling_active:
                    return func(*args, **kwargs)
                label = label_default or func.__name__
                token = _range_push(label, color, domain)
                try:
                    return func(*args, **kwargs)
                finally:
                    _range_pop(token)

            return sync_wrapper

        return decorator
