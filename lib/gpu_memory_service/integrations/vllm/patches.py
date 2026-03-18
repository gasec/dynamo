# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS vLLM patches and shadow mode utilities.

Patches are applied at GMSWorker import time. Shadow mode behaviour is gated
by the DYN_GMS_SHADOW_MODE env var and the model_runner._shadow_init_phase flag.
"""

from __future__ import annotations

import logging
import os
import time

import torch

from gpu_memory_service import get_gms_client_memory_manager
from gpu_memory_service.common.types import GrantedLockType

logger = logging.getLogger(__name__)

# Patch state tracking (to prevent double-patching)
_memory_snapshot_patched = False
_request_memory_patched = False
_register_kv_caches_patched = False
_initialize_kv_cache_tensors_patched = False
_get_slot_mappings_patched = False
_allocate_kv_cache_on_wake_added = False


# =============================================================================
# Shadow mode utilities
# =============================================================================


def is_shadow_mode() -> bool:
    """True when DYN_GMS_SHADOW_MODE=1 (set by main.py at startup)."""
    return os.environ.get("DYN_GMS_SHADOW_MODE", "0") == "1"


def force_piecewise_cudagraph_mode(engine_args) -> None:
    """Ensure PIECEWISE cudagraph mode for shadow engines.

    Shadow mode stubs attention during graph capture so no KV cache is
    needed. Raises if the user explicitly set a conflicting mode.
    """
    from vllm.config import CompilationConfig, CUDAGraphMode

    cc = engine_args.compilation_config
    assert isinstance(cc, CompilationConfig), (
        f"Expected CompilationConfig, got {type(cc).__name__}. "
        f"vLLM's arg parsing may have changed."
    )
    if cc.cudagraph_mode is None:
        cc.cudagraph_mode = CUDAGraphMode.PIECEWISE
    elif cc.cudagraph_mode != CUDAGraphMode.PIECEWISE:
        raise ValueError(
            f"Shadow mode requires PIECEWISE cudagraph mode, "
            f"got {cc.cudagraph_mode.name}"
        )
    logger.info("[Shadow] cudagraph_mode set to PIECEWISE")


# =============================================================================
# Core GMS patches (always applied)
# =============================================================================


def patch_memory_snapshot() -> None:
    """Add committed GMS bytes to MemorySnapshot.free_memory (RO mode only)."""
    global _memory_snapshot_patched

    if _memory_snapshot_patched:
        return

    try:
        from vllm.utils.mem_utils import MemorySnapshot
    except ImportError:
        logger.debug("[GMS Patch] MemorySnapshot not available")
        return

    original_measure = MemorySnapshot.measure

    def patched_measure(self):
        original_measure(self)

        manager = get_gms_client_memory_manager()
        assert manager is not None, "GMS client is not initialized"

        if manager.granted_lock_type == GrantedLockType.RO:
            allocations = manager.list_handles()
            committed_bytes = sum(alloc.get("aligned_size", 0) for alloc in allocations)
        else:
            committed_bytes = 0
            logger.info("[GMS] RW mode - skipping committed memory adjustment")

        original_free = self.free_memory
        self.free_memory += committed_bytes

        if committed_bytes > 0:
            logger.info(
                "[GMS Patch] Adjusted free_memory: %.2f GiB + %.2f GiB = %.2f GiB",
                original_free / (1 << 30),
                committed_bytes / (1 << 30),
                self.free_memory / (1 << 30),
            )

    MemorySnapshot.measure = patched_measure
    _memory_snapshot_patched = True
    logger.info("[GMS Patch] Patched MemorySnapshot.measure")


# =============================================================================
# Shadow mode patches
# =============================================================================


def patch_request_memory() -> None:
    """Bypass free >= requested check (shadow shares GPU with active engine)."""
    global _request_memory_patched

    if _request_memory_patched:
        return

    try:
        from vllm.v1.worker import utils as worker_utils
    except ImportError:
        logger.debug("[GMS Patch] vllm.v1.worker.utils not available")
        return

    original_request_memory = worker_utils.request_memory

    def patched_request_memory(init_snapshot, cache_config):
        if is_shadow_mode():
            requested_memory = int(
                init_snapshot.total_memory * cache_config.gpu_memory_utilization
            )
            logger.info(
                "[GMS Patch] Shadow mode: bypassing memory check "
                "(requested=%.2f GiB, free=%.2f GiB)",
                requested_memory / (1 << 30),
                init_snapshot.free_memory / (1 << 30),
            )
            return requested_memory

        return original_request_memory(init_snapshot, cache_config)

    worker_utils.request_memory = patched_request_memory
    _request_memory_patched = True
    logger.info("[GMS Patch] Patched request_memory for shadow mode support")


def patch_determine_available_memory() -> None:
    """Project available memory from total GPU capacity (shadow shares GPU)."""
    if not is_shadow_mode():
        return

    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:
        logger.debug("[GMS Patch] Worker not available")
        return

    original_determine = Worker.determine_available_memory

    def patched_determine_available_memory(self):
        # Run profile for torch.compile; measure peak to get non-KV usage.
        # max_memory_allocated() = weights + activations + buffers (PyTorch-managed).
        torch.cuda.reset_peak_memory_stats()
        self.model_runner.profile_run()
        torch.cuda.synchronize()
        non_kv_cache_memory = torch.cuda.max_memory_allocated()

        projected_available = self.requested_memory - non_kv_cache_memory

        logger.info(
            "[GMS Patch] Shadow mode: projected available memory "
            "%.2f GiB (requested=%.2f GiB, non_kv=%.2f GiB)",
            projected_available / (1 << 30),
            self.requested_memory / (1 << 30),
            non_kv_cache_memory / (1 << 30),
        )

        return int(projected_available)

    Worker.determine_available_memory = patched_determine_available_memory
    logger.info(
        "[GMS Patch] Patched determine_available_memory for shadow mode"
    )


def patch_register_kv_caches() -> None:
    """Skip NixlConnector.register_kv_caches when kv_caches is empty."""
    global _register_kv_caches_patched

    if _register_kv_caches_patched:
        return

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
            NixlConnector,
        )
    except ImportError:
        logger.debug("[GMS Patch] NixlConnector not available")
        return

    original_register = NixlConnector.register_kv_caches

    def patched_register_kv_caches(self, kv_caches):
        if not kv_caches:
            logger.info("[GMS Patch] Skipping KV cache registration (empty kv_caches)")
            return
        return original_register(self, kv_caches)

    NixlConnector.register_kv_caches = patched_register_kv_caches
    _register_kv_caches_patched = True
    logger.info("[GMS Patch] Patched NixlConnector.register_kv_caches")


def patch_initialize_kv_cache_tensors() -> None:
    """No-op during shadow init; store config for later allocation on wake."""
    global _initialize_kv_cache_tensors_patched

    if _initialize_kv_cache_tensors_patched:
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.debug("[GMS Patch] GPUModelRunner not available")
        return

    original_initialize_kv_cache_tensors = GPUModelRunner.initialize_kv_cache_tensors

    def patched_initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        if getattr(self, "_shadow_init_phase", False):
            self._shadow_kv_cache_config = kv_cache_config
            self._shadow_kernel_block_sizes = kernel_block_sizes
            logger.info(
                "[Shadow] Init phase: stored config, skipping KV cache allocation"
            )
            return {}

        return original_initialize_kv_cache_tensors(
            self, kv_cache_config, kernel_block_sizes
        )

    GPUModelRunner.initialize_kv_cache_tensors = patched_initialize_kv_cache_tensors
    _initialize_kv_cache_tensors_patched = True
    logger.info("[GMS Patch] Patched GPUModelRunner.initialize_kv_cache_tensors")


def patch_get_slot_mappings() -> None:
    """Return (None, None) when KV caches are empty.

    _dummy_run() calls _get_slot_mappings() unconditionally during warmup.
    Without KV tensors there is nothing to index into; returning (None, None)
    makes KV write ops gracefully no-op.
    """
    global _get_slot_mappings_patched

    if _get_slot_mappings_patched:
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.debug("[GMS Patch] GPUModelRunner not available")
        return

    original_get_slot_mappings = GPUModelRunner._get_slot_mappings

    def patched_get_slot_mappings(self, *args, **kwargs):
        if not self.kv_caches:
            return None, None
        return original_get_slot_mappings(self, *args, **kwargs)

    GPUModelRunner._get_slot_mappings = patched_get_slot_mappings
    _get_slot_mappings_patched = True
    logger.info("[GMS Patch] Patched GPUModelRunner._get_slot_mappings")


def patch_allocate_kv_cache_on_wake() -> None:
    """Add allocate_kv_cache_on_wake to GPUModelRunner.

    Called by GMSWorker.wake_up() after _shadow_init_phase is cleared.
    Waits for GPU memory to be freed (60 s timeout), then allocates.
    """
    global _allocate_kv_cache_on_wake_added

    if _allocate_kv_cache_on_wake_added:
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.debug("[GMS Patch] GPUModelRunner not available")
        return

    if hasattr(GPUModelRunner, "allocate_kv_cache_on_wake"):
        logger.debug("[GMS Patch] allocate_kv_cache_on_wake already exists")
        return

    def allocate_kv_cache_on_wake(self) -> dict:
        assert hasattr(self, "_shadow_kv_cache_config"), (
            "_shadow_kv_cache_config not set"
        )
        assert hasattr(self, "_shadow_kernel_block_sizes"), (
            "_shadow_kernel_block_sizes not set"
        )

        config = self._shadow_kv_cache_config
        kv_cache_bytes = sum(t.size for t in config.kv_cache_tensors)

        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < kv_cache_bytes:
            logger.info(
                "[Shadow] Waiting for GPU memory (need %.2f GiB, free %.2f GiB)",
                kv_cache_bytes / (1 << 30),
                free_bytes / (1 << 30),
            )
            deadline = time.monotonic() + 60.0
            while free_bytes < kv_cache_bytes:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Timed out waiting for GPU memory: "
                        f"need {kv_cache_bytes / (1 << 30):.2f} GiB, "
                        f"free {free_bytes / (1 << 30):.2f} GiB"
                    )
                time.sleep(0.5)
                free_bytes = torch.cuda.mem_get_info()[0]
            logger.info(
                "[Shadow] GPU memory available (free %.2f GiB), proceeding",
                free_bytes / (1 << 30),
            )

        logger.info("[Shadow] Allocating KV cache on wake")

        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(self.vllm_config):
            kv_caches = self.initialize_kv_cache_tensors(
                config,
                self._shadow_kernel_block_sizes,
            )

        # Re-register with KV transfer group (skipped at init since kv_caches was {}).
        # Mirrors GPUModelRunner.initialize_kv_cache() — update if upstream changes.
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.base import (
                get_kv_transfer_group,
                has_kv_transfer_group,
            )

            if has_kv_transfer_group() and kv_caches:
                kv_transfer_group = get_kv_transfer_group()
                kv_transfer_group.register_kv_caches(kv_caches)
                logger.debug("[Shadow] Registered KV caches with transfer group")
        except ImportError:
            logger.debug("[Shadow] KV transfer group not available")

        total_bytes = sum(t.numel() * t.element_size() for t in kv_caches.values())
        logger.info(
            "[Shadow] Allocated KV cache on wake: %.2f GiB (%d tensors)",
            total_bytes / (1 << 30),
            len(kv_caches),
        )

        return kv_caches

    GPUModelRunner.allocate_kv_cache_on_wake = allocate_kv_cache_on_wake
    _allocate_kv_cache_on_wake_added = True
    logger.info("[GMS Patch] Added GPUModelRunner.allocate_kv_cache_on_wake")


def patch_cudagraph_mode_escalation() -> None:
    """Clamp cudagraph_mode to PIECEWISE if vLLM escalates to FULL_AND_PIECEWISE."""
    if not is_shadow_mode():
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        return

    if getattr(GPUModelRunner, "_shadow_cg_escalation_patched", False):
        return

    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    original_init_keys = CudagraphDispatcher.initialize_cudagraph_keys

    def patched_initialize_cudagraph_keys(self, cudagraph_mode, *args, **kwargs):
        if is_shadow_mode():
            from vllm.config import CUDAGraphMode

            if cudagraph_mode not in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE):
                logger.info(
                    "[Shadow] Clamping cudagraph_mode from %s to PIECEWISE",
                    cudagraph_mode.name,
                )
                cudagraph_mode = CUDAGraphMode.PIECEWISE
        return original_init_keys(self, cudagraph_mode, *args, **kwargs)

    CudagraphDispatcher.initialize_cudagraph_keys = patched_initialize_cudagraph_keys
    GPUModelRunner._shadow_cg_escalation_patched = True
    logger.info("[GMS Patch] Patched cudagraph mode escalation for shadow mode")


# =============================================================================
# Patch application helper
# =============================================================================


def apply_shadow_mode_patches() -> None:
    """Apply all shadow mode patches (safe for non-shadow engines)."""
    patch_request_memory()
    patch_determine_available_memory()
    patch_register_kv_caches()
    patch_initialize_kv_cache_tensors()
    patch_get_slot_mappings()
    patch_allocate_kv_cache_on_wake()
    patch_cudagraph_mode_escalation()
    logger.info("[GMS Patch] Shadow mode patches applied")
