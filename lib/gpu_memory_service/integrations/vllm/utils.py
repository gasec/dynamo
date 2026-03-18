# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shadow mode utilities for GMS vLLM integration."""

import logging
import os

logger = logging.getLogger(__name__)


def is_shadow_mode() -> bool:
    """True when DYN_GMS_SHADOW_MODE=1 (set by main.py at startup)."""
    return os.environ.get("DYN_GMS_SHADOW_MODE", "0") == "1"


def validate_cudagraph_mode(engine_args) -> None:
    """Validate and set cudagraph mode for shadow engines.

    Defaults unset mode to PIECEWISE (attention stubbed during graph capture).
    Accepts NONE (e.g. enforce_eager). Rejects FULL variants which need
    KV cache tensors that don't exist during shadow init.
    """
    from vllm.config import CompilationConfig, CUDAGraphMode

    cc = engine_args.compilation_config
    assert isinstance(cc, CompilationConfig), (
        f"Expected CompilationConfig, got {type(cc).__name__}. "
        f"vLLM's arg parsing may have changed."
    )
    if cc.cudagraph_mode is None:
        cc.cudagraph_mode = CUDAGraphMode.PIECEWISE
        logger.info("[Shadow] cudagraph_mode defaulted to PIECEWISE")
    elif cc.cudagraph_mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE):
        pass  # compatible
    else:
        raise ValueError(
            f"Shadow mode requires PIECEWISE or NONE cudagraph mode, "
            f"got {cc.cudagraph_mode.name}. FULL modes capture attention ops "
            f"that need KV cache tensors, which don't exist during shadow init."
        )
