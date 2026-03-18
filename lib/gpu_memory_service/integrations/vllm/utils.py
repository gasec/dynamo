# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shadow mode utilities for GMS vLLM integration."""

import logging
import os

logger = logging.getLogger(__name__)


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
