# PR #6818 Review Ledger

## 1. Replace `--gms-mode` enum with `--gms-shadow-mode` boolean flag

**Files**: `backend_args.py`, `args.py`, `main.py`

**Current**: `--gms-mode` string enum (`normal|shadow`), `DYN_VLLM_GMS_MODE` env var, `gms_mode` config field.

**Change to**: `--gms-shadow-mode` boolean flag, `DYN_VLLM_GMS_SHADOW_MODE` env var.

**Why**: Only two states — shadow or not. A string enum implies future modes that won't exist. A boolean flag is simpler API surface and reads more naturally in validation ("shadow mode requires GMS" vs "gms-mode shadow requires gms load format").

## 2. Dedup `sleep_engine`/`wake_engine` with HTTP handlers

**File**: `handlers.py`

**Current**: `sleep_engine`/`wake_engine` duplicate the GPU ops (`pause_generation`/`sleep`/`wake_up`/`resume_generation`) that also appear in the `sleep`/`wake_up` HTTP handlers.

**Change to**: Have the HTTP handlers call `sleep_engine`/`wake_engine` for the GPU portion, and only add discovery + lock + state tracking around them.

**Why**: Removes duplicated GPU lifecycle calls. The shadow init sequence in `main.py` correctly uses `sleep_engine`/`wake_engine` directly (no discovery side effects), while the HTTP handlers compose on top of them.

## 3. Simplify and harden PIECEWISE cudagraph forcing

**Files**: `main.py`, `worker.py`, new shared utility

**Current**: 4-branch isinstance chain (None/dict/str/CompilationConfig) duplicated in both `setup_vllm_engine` and `run_dynamo_headless`. Headless path reads `SHADOW_SKIP_KV_CACHE` env var instead of config. Silent override of user's explicit cudagraph mode. Headless path silently swallows JSONDecodeError.

**Changes**:

a) **Simplify the force logic**. Analysis of vLLM's arg parsing (`EngineArgs.__post_init__`, `_compute_kwargs`) confirms `compilation_config` is always a `CompilationConfig` object by the time we see it — never None/dict/str. The isinstance chain is dead code. Replace with:
```python
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
```

b) **Extract to shared utility** in a GMS/vLLM-specific location. Both `setup_vllm_engine` and `run_dynamo_headless` call the same function.

c) **Headless path reads config, not env var**. `run_dynamo_headless` has the full `Config` object — use `config.gms_shadow_mode` instead of `os.environ.get("SHADOW_SKIP_KV_CACHE")`. Nest under the `load_format == "gms"` check for consistency.

d) **Post-hoc assertion in `GMSWorker.initialize_from_config()`**. vLLM mutates `cudagraph_mode` in multiple places after our code runs:
  - `CompilationConfig.__post_init__` (can downgrade PIECEWISE → NONE)
  - `VllmConfig.__post_init__` (enforce_eager, model type overrides)
  - `GPUModelRunner._check_and_update_cudagraph_mode` (attention backend resolution)

  Override `initialize_from_config` in `GMSWorker` to assert PIECEWISE after all mutations:
  ```python
  def initialize_from_config(self, kv_cache_config):
      super().initialize_from_config(kv_cache_config)
      if is_shadow_mode():
          mode = self.model_runner.compilation_config.cudagraph_mode
          if mode != CUDAGraphMode.PIECEWISE:
              raise RuntimeError(
                  f"Shadow mode requires PIECEWISE cudagraph mode after resolution, "
                  f"but got {mode.name}. vLLM's config resolution overrode it."
              )
  ```
  Works for single-node TP>1 (all workers are GMSWorker) and headless multi-node (uses GMSWorker after worker_cls propagation fix).

e) **Rename env var**. `SHADOW_SKIP_KV_CACHE` describes implementation, not intent. Rename to `DYN_GMS_SHADOW_MODE` for consistency with CLI flag. Wrap in a shared `is_shadow_mode()` utility (promote existing `_is_shadow_mode()` from `patches.py`).

**Why**: Eliminates dead code, fails loud on bad assumptions, catches downstream mutations, removes duplication, and makes the env var / config propagation consistent.

## 4. `patch_request_memory` — use shared shadow mode check, simplify comments

**File**: `patches.py`

**Current**: Uses `_is_shadow_mode()` (private, checks `SHADOW_SKIP_KV_CACHE` env var). Docstring has a "Note:" explaining why it checks env var instead of `_shadow_init_phase`.

**Change to**: Use the shared `is_shadow_mode()` utility (from item 3e). Remove the "Note:" comment — once we have a proper `is_shadow_mode()` utility, the distinction doesn't need explaining. Simplify surrounding comments to be terse.

**Why**: Consistent with the `is_shadow_mode()` utility introduced in item 3e. The implementation detail about *which* check is used is no longer noteworthy once there's a single canonical check.

## 5. `patch_determine_available_memory` — simplify peak memory calculation

**File**: `patches.py`

**Current**: Decomposes peak memory into `model_bytes` + `activation_bytes`, then recombines as `non_kv_cache_memory = model_bytes + activation_bytes`. The decomposition is redundant — it equals `peak_bytes`. Log line separately reports weights and activations.

**Change to**:
```python
# max_memory_allocated() returns the high-water mark of GPU memory
# managed by PyTorch's caching allocator (in bytes) since the last
# reset_peak_memory_stats() call. This includes all live tensors
# (weights, activations, buffers) but excludes non-PyTorch allocations
# (CUDA contexts, cuBLAS workspaces, NCCL buffers).
torch.cuda.reset_peak_memory_stats()
self.model_runner.profile_run()
torch.cuda.synchronize()
non_kv_cache_memory = torch.cuda.max_memory_allocated()

projected_available = self.requested_memory - non_kv_cache_memory
```
Update log line to report `non_kv_cache_memory` as a single value instead of split weights/activations. Also use shared `is_shadow_mode()` utility.

**Why**: The decomposition round-trips to `peak_bytes` anyway. Keeping it together is simpler and more honest — we don't have a precise split (non-torch overhead is not captured either way).

## 6. `patch_get_slot_mappings` — rewrite docstring

**File**: `patches.py`

**Current**: Docstring references "vLLM v0.15.x" and `unified_kv_cache_update`, and explains the downstream effect without explaining why the call happens in the first place.

**Change to**:
```python
def patch_get_slot_mappings() -> None:
    """Patch GPUModelRunner._get_slot_mappings to return (None, None) when KV caches are empty.

    Slot mappings translate logical token positions to physical KV cache addresses.
    _dummy_run() (used for CUDA graph capture/warmup) calls _get_slot_mappings()
    unconditionally — even in PIECEWISE mode where attention ops are excluded from
    the captured graph. Normally harmless since the KV cache exists, but in shadow
    mode there are no KV cache tensors or block tables to index into.

    Returning (None, None) causes set_forward_context to default slot_mapping to {},
    which makes KV write ops gracefully no-op (they check slot_mapping.get(layer_name)
    is not None before writing).
    """
```
Also use shared `is_shadow_mode()` utility.

## 7. `patch_allocate_kv_cache_on_wake` — replace 70% barrier with exact size, add timeout

**File**: `patches.py`

**Current**: Hardcoded `needed_bytes = int(0.7 * total_bytes)` barrier with no timeout on the poll loop.

**Change to**:
```python
config = self._shadow_kv_cache_config
kv_cache_bytes = sum(t.size for t in config.kv_cache_tensors)

free_bytes, _ = torch.cuda.mem_get_info()
if free_bytes < kv_cache_bytes:
    logger.info(
        "[Shadow] Waiting for GPU memory before KV cache allocation "
        "(need %.2f GiB, free %.2f GiB)",
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
```

**Why**: `KVCacheConfig.kv_cache_tensors[*].size` gives the exact byte count for each tensor that `initialize_kv_cache_tensors` will allocate via `torch.zeros(size, dtype=torch.int8)`. No reason to guess with a magic 70% constant when we have the exact number. Timeout prevents infinite spin if the primary engine doesn't release memory (e.g., hung process, OOM kill delayed).

Also: move `import time` / `import torch` to module level, and add a comment noting that `_shadow_init_phase` is cleared by the caller (`GMSWorker.wake_up`).

Add a comment on the KV transfer group registration block explaining:
```python
# Re-register KV caches with the KV transfer group (NIXL / disaggregated
# prefill-decode). This was skipped at init because kv_caches was {}.
# Mirrors the registration in GPUModelRunner.initialize_kv_cache()
# (gpu_model_runner.py ~L6115-6124) — if that upstream code changes
# (e.g., adds set_host_xfer_buffer_ops or register_cross_layers_kv_cache),
# this path must be updated to match.
```

## 8. Refactor patches into subclasses

**See**: [`patches-refactor-plan.md`](patches-refactor-plan.md)

Replace 5 of 8 monkey-patches with method overrides on `GMSWorker` (existing) and `GMSModelRunner` (new subclass of `GPUModelRunner`). Consolidates `determine_available_memory`, `initialize_kv_cache_tensors`, `_get_slot_mappings`, `_check_and_update_cudagraph_mode`, and `allocate_kv_cache_on_wake` into clean class overrides. 3 patches remain as monkey-patches (`memory_snapshot`, `request_memory`, `register_kv_caches`). Adds post-hoc cudagraph assertion in `GMSWorker.initialize_from_config()` as safety net.
