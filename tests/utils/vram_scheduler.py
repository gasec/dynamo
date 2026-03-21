# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU VRAM utilities for parallel test execution.

Functions:
    detect_gpus()                  Enumerate GPUs via pynvml
    auto_worker_count(gpus, limit) Calculate slot count for -n auto
    write_test_meta(items)         Serialize profiled/requested vram + timeout
    load_test_meta()               Read the serialized test metadata
    print_gpu_plan(gpus, limit, would_run)  Dry-run GPU plan summary

Usage:
    # Sequential (filter only)
    pytest --max-vram-gib=10 -m "gpu_1 and vllm" tests/serve/

    # Parallel (VRAM-aware scheduling)
    pytest --max-vram-gib=10 -n auto -m "gpu_1 and vllm" tests/serve/
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import pynvml

_logger = logging.getLogger(__name__)

# Reserve 10% of GPU VRAM for CUDA contexts, driver overhead, etc.
VRAM_SAFETY_FACTOR = 0.90

_TEST_META_FILENAME = "pytest_gpu_parallel_test_meta.json"


def detect_gpus() -> list[dict]:
    """Return list of dicts with 'index', 'name', 'total_mib' per GPU.

    Uses pynvml (already a dependency via profile_pytest.py).
    Returns empty list if no GPUs or pynvml is unavailable.
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return []
    try:
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "total_mib": mem.total // (1024 * 1024),
                }
            )
        return gpus
    finally:
        pynvml.nvmlShutdown()


def auto_worker_count(
    gpus: list[dict],
    vram_limit: float,
    test_profiled_gibs: list[float] | None = None,
) -> int:
    """Calculate slot count for -n auto.

    Uses the smallest profiled test size (if provided) to maximize parallelism.
    Falls back to vram_limit when no test sizes are available.
    """
    if not gpus or vram_limit <= 0:
        return len(gpus) or 1
    min_gpu_gib = min(g["total_mib"] for g in gpus) / 1024.0
    budget_gib = min_gpu_gib * VRAM_SAFETY_FACTOR
    divisor = vram_limit
    if test_profiled_gibs:
        divisor = min(test_profiled_gibs)
    workers_per_gpu = max(1, int(budget_gib / divisor))
    return len(gpus) * workers_per_gpu


def write_test_meta(items, dest_dir: str | None = None) -> None:
    """Serialize profiled_vram_gib, requested_vram_gib, and timeout markers to JSON.

    Called from pytest_collection_modifyitems so the GPU orchestrator can
    read test metadata without re-collecting.
    """
    test_meta: dict[str, dict] = {}
    for item in items:
        meta: dict = {}
        profiled_mark = item.get_closest_marker("profiled_vram_gib")
        if profiled_mark and profiled_mark.args:
            meta["profiled_vram_gib"] = profiled_mark.args[0]
        requested_mark = item.get_closest_marker("requested_vram_gib")
        if requested_mark and requested_mark.args:
            meta["requested_vram_gib"] = requested_mark.args[0]
        timeout_mark = item.get_closest_marker("timeout")
        if timeout_mark and timeout_mark.args:
            meta["timeout"] = timeout_mark.args[0]
        if meta:
            test_meta[item.nodeid] = meta
    if test_meta:
        path = os.path.join(dest_dir or tempfile.gettempdir(), _TEST_META_FILENAME)
        with open(path, "w") as f:
            json.dump(test_meta, f)


def load_test_meta() -> dict[str, dict]:
    """Load the nodeid -> {profiled_vram_gib, requested_vram_gib, timeout} map."""
    path = os.path.join(tempfile.gettempdir(), _TEST_META_FILENAME)
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def print_gpu_plan(
    gpus: list[dict], vram_limit: float, would_run: list[tuple[str, float]]
) -> None:
    """Print the GPU-parallel plan section for --dry-run output."""
    min_gpu_gib = min(g["total_mib"] for g in gpus) / 1024.0
    budget_gib = min_gpu_gib * VRAM_SAFETY_FACTOR
    profiled_gibs = [gib for _, gib in would_run if gib is not None]
    min_test_gib = min(profiled_gibs) if profiled_gibs else vram_limit
    auto_slots = max(1, int(budget_gib / min_test_gib))

    print(f"\n{'=' * 60}")
    print("GPU-Parallel Plan")
    print(f"{'=' * 60}")
    for gpu in gpus:
        gib = gpu["total_mib"] / 1024
        print(f"  GPU {gpu['index']}: {gpu['name']} ({gib:.1f} GiB)")
    print(f"\n  Usable VRAM: {budget_gib:.0f} GiB (after 10% safety)")
    print("\n  Run options:")
    print("    (no -n)  : sequential, 1 test at a time")
    print(
        f"    -n auto  : up to {auto_slots} slots per GPU "
        f"({budget_gib:.0f} / {min_test_gib:.0f} GiB smallest test)"
    )
    print(f"    -n N     : N concurrent slots across {len(gpus)} GPU(s)")
    print("\n  Usage:")
    print(
        f"    python tests/utils/pytest_parallel_gpu.py "
        f"--max-vram-gib={vram_limit:.0f} -n {auto_slots} "
        f"<pytest args>"
    )
