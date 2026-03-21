#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared GPU utility functions for launch scripts (source, don't execute).
#
# Usage:
#   source "$(dirname "$(readlink -f "$0")")/../common/gpu_utils.sh"
#   # or with SCRIPT_DIR already set:
#   source "$SCRIPT_DIR/../common/gpu_utils.sh"
#
# Functions (all return via stdout):
#   build_gpu_mem_args <engine> --model <name> [options...]
#       Returns GPU memory fraction from _PROFILE_PYTEST_VRAM_FRAC_OVERRIDE
#       or user flag; empty if neither set (engine uses its own default).
#   gpu_gb_to_total_fraction <gib>    Fraction of TOTAL VRAM (vLLM/sglang)
#   gpu_gb_to_free_fraction <gib>     Fraction of FREE VRAM (TensorRT-LLM)

# build_gpu_mem_args <engine> --model <name> [options...]
#
# Prints the GPU memory fraction to stdout (empty line if none).
# Callers capture with:  GPU_MEM_FRACTION=$(build_gpu_mem_args ...)
#
# Priority:
#   1. _PROFILE_PYTEST_VRAM_FRAC_OVERRIDE  (from pytest requested_vram_gib marker)
#   2. User flag (--gpu-memory-utilization / --mem-fraction-static)
#   3. Empty  (let engine use its own default, e.g. vLLM 0.90)
#
# Options:
#   --model NAME                 Model name (required; aliases: --model-path)
#   --gpu-memory-utilization F   User override (aliases: --mem-fraction-static)
#   --workers-per-gpu N          Divide the fraction by N (for shared-GPU disagg)
#
# Usage:
#   GPU_MEM_FRACTION=$(build_gpu_mem_args vllm --model "$MODEL")
#   python -m dynamo.vllm --model "$MODEL" \
#       ${GPU_MEM_FRACTION:+--gpu-memory-utilization "$GPU_MEM_FRACTION"} &
build_gpu_mem_args() {
    local engine="${1:?usage: build_gpu_mem_args <engine> --model <name> [options...]}"
    shift

    local model=""
    local workers_per_gpu=1
    local user_frac=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model|--model-path)
                                model="$2";           shift 2 ;;
            --gpu-memory-utilization|--mem-fraction-static)
                                user_frac="$2";       shift 2 ;;
            --workers-per-gpu)  workers_per_gpu="$2"; shift 2 ;;
            --max-model-len|--context-length|--max-seq-len|\
            --max-num-seqs|--max-running-requests|--max-batch-size)
                                shift 2 ;;  # accepted but ignored (legacy compat)
            *) echo "build_gpu_mem_args: unknown option '$1'" >&2; return 1 ;;
        esac
    done

    if [[ -z "$model" ]]; then
        echo "build_gpu_mem_args: --model is required" >&2
        return 1
    fi

    # Priority:
    #   1. _PROFILE_PYTEST_VRAM_FRAC_OVERRIDE (from pytest requested_vram_gib marker)
    #   2. User flag (--gpu-memory-utilization / --mem-fraction-static)
    #   3. Empty (engine uses its own default, e.g. vLLM 0.90)
    local frac=""
    if [[ -n "${_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE:-}" ]]; then
        frac="$_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE"
    elif [[ -n "$user_frac" ]]; then
        frac="$user_frac"
    fi

    # --workers-per-gpu divides the fraction evenly
    if [[ -n "$frac" && "$workers_per_gpu" -gt 1 ]]; then
        frac=$(awk -v f="$frac" -v n="$workers_per_gpu" 'BEGIN { printf "%.2f", f / n }')
    fi

    echo "$frac"
}


# gpu_gb_to_total_fraction <gib> [gpu_index]
#
# For vLLM / sglang: --gpu-memory-utilization is a fraction of TOTAL GPU memory.
# The engine budgets model weights + KV cache + activations within that limit.
#
# Prints the fraction of total GPU VRAM that <gib> GiB represents.
# Useful for converting portable absolute memory requirements to
# engine-specific fraction parameters (--gpu-memory-utilization, etc).
#
# Examples:
#   gpu_gb_to_total_fraction 4        # on 48 GiB GPU → 0.09
#   gpu_gb_to_total_fraction 16       # on 48 GiB GPU → 0.34
#   gpu_gb_to_total_fraction 4 1      # query GPU index 1 instead of 0
#
# The result is ceil-rounded to 2 decimal places with a minimum of 0.05
# and a maximum of 0.95.
gpu_gb_to_total_fraction() {
    local gib=${1:?usage: gpu_gb_to_total_fraction <gib> [gpu_index]}
    local gpu_idx=${2:-0}

    local total_mib
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null)
    if [[ -z "$total_mib" || "$total_mib" -eq 0 ]]; then
        echo "gpu_gb_to_total_fraction: failed to query GPU $gpu_idx total memory" >&2
        return 1
    fi

    local total_gib
    total_gib=$(awk -v t="$total_mib" 'BEGIN { printf "%.1f", t / 1024 }')

    if awk -v gib="$gib" -v total="$total_mib" 'BEGIN { exit (gib * 1024 > total) ? 0 : 1 }'; then
        echo "" >&2
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
        echo "WARNING: Requested ${gib} GiB but GPU $gpu_idx only has ${total_gib} GiB total." >&2
        echo "The model likely won't fit. Consider a GPU with more VRAM" >&2
        echo "or reduce the model size (quantization, smaller model, etc)." >&2
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
        echo "" >&2
    fi

    # fraction = gib * 1024 / total_mib, ceil to 2 decimals, clamp [0.05, 0.95]
    awk -v gib="$gib" -v total="$total_mib" 'BEGIN {
        frac = (gib * 1024) / total
        # ceil to 2 decimal places
        frac = int(frac * 100 + 0.99) / 100
        if (frac < 0.05) frac = 0.05
        if (frac > 0.95) frac = 0.95
        printf "%.2f\n", frac
    }'
}

# gpu_gb_to_free_fraction <gib> [gpu_index]
#
# For TensorRT-LLM: fraction of FREE memory (after model load), not total.
# Queries current free memory via nvidia-smi and returns gib / free_gib.
# Clamped [0.01, 0.95], ceil-rounded to 2 decimal places.
gpu_gb_to_free_fraction() {
    local gib=${1:?usage: gpu_gb_to_free_fraction <gib> [gpu_index]}
    local gpu_idx=${2:-0}

    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null)
    if [[ -z "$free_mib" || "$free_mib" -eq 0 ]]; then
        echo "gpu_gb_to_free_fraction: failed to query GPU $gpu_idx free memory" >&2
        return 1
    fi

    local free_gib
    free_gib=$(awk -v f="$free_mib" 'BEGIN { printf "%.1f", f / 1024 }')

    if awk -v gib="$gib" -v free="$free_mib" 'BEGIN { exit (gib * 1024 > free) ? 0 : 1 }'; then
        echo "" >&2
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
        echo "WARNING: Requested ${gib} GiB KV cache but GPU $gpu_idx only has ${free_gib} GiB free." >&2
        echo "After model loading, even less will be available." >&2
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
        echo "" >&2
    fi

    # fraction = gib * 1024 / free_mib, ceil to 2 decimals, clamp [0.01, 0.95]
    awk -v gib="$gib" -v free="$free_mib" 'BEGIN {
        frac = (gib * 1024) / free
        frac = int(frac * 100 + 0.99) / 100
        if (frac < 0.01) frac = 0.01
        if (frac > 0.95) frac = 0.95
        printf "%.2f\n", frac
    }'
}

# ---------------------------------------------------------------------------
# Self-test: bash gpu_utils.sh --self-test
# ---------------------------------------------------------------------------
_gpu_utils_self_test() {
    local pass=0 fail=0
    _assert() {
        local label="$1" expected="$2" actual="$3"
        if [[ "$expected" == "$actual" ]]; then
            ((pass++))
            echo "  PASS  $label"
        else
            ((fail++))
            echo "  FAIL  $label  (expected='$expected'  actual='$actual')"
        fi
    }

    echo "=== build_gpu_mem_args: profiler override wins ==="

    local frac
    frac=$(_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE=0.55 \
        build_gpu_mem_args vllm --model "Qwen/Qwen3-0.6B" --gpu-memory-utilization 0.70)
    _assert "profiler beats user flag" "0.55" "$frac"

    echo ""
    echo "=== build_gpu_mem_args: user flag ==="

    frac=$(build_gpu_mem_args vllm --model "Qwen/Qwen3-0.6B" --gpu-memory-utilization 0.70)
    _assert "user flag" "0.70" "$frac"

    echo ""
    echo "=== build_gpu_mem_args: no override = empty ==="

    frac=$(build_gpu_mem_args vllm --model "Qwen/Qwen3-0.6B")
    _assert "empty (engine default)" "" "$frac"

    echo ""
    echo "=== build_gpu_mem_args: --workers-per-gpu divides ==="

    frac=$(_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE=0.80 \
        build_gpu_mem_args vllm --model "Qwen/Qwen3-0.6B" --workers-per-gpu 2)
    _assert "0.80/2 = 0.40" "0.40" "$frac"

    echo ""
    echo "=== build_gpu_mem_args: sglang flag ==="

    frac=$(build_gpu_mem_args sglang --model-path "Qwen/Qwen3-0.6B" --mem-fraction-static 0.60)
    _assert "sglang user flag" "0.60" "$frac"

    echo ""
    echo "=== build_gpu_mem_args: missing --model ==="

    build_gpu_mem_args vllm 2>/dev/null
    _assert "missing --model exits 1" "1" "$?"

    echo ""
    echo "=== build_gpu_mem_args: legacy flags ignored ==="

    frac=$(_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE=0.50 \
        build_gpu_mem_args vllm --model "Qwen/Qwen3-0.6B" --max-model-len 4096 --max-num-seqs 2)
    _assert "legacy flags accepted" "0.50" "$frac"

    echo ""
    echo "=========================================="
    echo "Results: $pass passed, $fail failed"
    echo "=========================================="
    [[ "$fail" -eq 0 ]]
}

# Self-test: source this file then call _gpu_utils_self_test
if [[ "${BASH_SOURCE[0]}" == "$0" && "${1:-}" == "--self-test" ]]; then
    _gpu_utils_self_test
    exit $?
fi
