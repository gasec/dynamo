# Patches Refactor: Monkey-patches → Subclasses

## Summary

Replace 5 of 8 monkey-patches in `patches.py` with method overrides on two subclasses:
`GMSWorker(Worker)` (existing) and `GMSModelRunner(GPUModelRunner)` (new). 3 patches remain
as monkey-patches because their targets have no class injection point.

## Injection mechanism

`GMSModelRunner` is injected via `__class__` swap in `GMSWorker.init_device()`:

```python
def init_device(self):
    super().init_device()  # constructs self.model_runner as GPUModelRunnerV1
    if not self.use_v2_model_runner:
        self.model_runner.__class__ = GMSModelRunner
```

Safe because `GMSModelRunner` only adds/overrides methods — no `__init__`, no `__slots__`,
no layout changes. `isinstance(runner, GPUModelRunner)` still passes.

Alternative: propose `model_runner_cls` class attribute upstream to vLLM (one-line change).

## What moves where

### GMSWorker (existing subclass of Worker)

| Method | Source | Notes |
|--------|--------|-------|
| `determine_available_memory()` | patch #3 | Direct override. Projects full GPU capacity in shadow mode using `max_memory_allocated()` |
| `initialize_from_config()` | new (ledger 3d) | Assert-only safety net after super(). Verifies cudagraph mode is PIECEWISE or NONE |

### GMSModelRunner (new subclass of GPUModelRunner)

| Method | Source | Notes |
|--------|--------|-------|
| `initialize_kv_cache_tensors()` | patch #5 | Returns `{}` during `_shadow_init_phase`, stores config for later |
| `_get_slot_mappings()` | patch #6 | Returns `(None, None)` when `self.kv_caches` is empty |
| `_check_and_update_cudagraph_mode()` | patch #8 | Forces PIECEWISE in shadow mode, calls `initialize_cudagraph_keys` once with correct mode |
| `allocate_kv_cache_on_wake()` | patch #7 | New method. Allocates KV cache on wake using stored config |

### Remaining monkey-patches (3)

| Patch | Target | Why it stays |
|-------|--------|-------------|
| `patch_memory_snapshot` | `MemorySnapshot.measure` | Dataclass instantiated everywhere in vLLM, no injection point |
| `patch_request_memory` | `request_memory()` | Free function in `worker_utils`, not a method |
| `patch_register_kv_caches` | `NixlConnector.register_kv_caches` | Call site buried inside `super().initialize_kv_cache()`, can't intercept without duplicating the 50-line method |

## Cudagraph mode: two-layer defense

The `_check_and_update_cudagraph_mode` override and `initialize_from_config` assertion
work together:

1. **`GMSModelRunner._check_and_update_cudagraph_mode()`** — prevents the wrong mode.
   In shadow mode, skips backend resolution entirely, forces PIECEWISE, calls
   `initialize_cudagraph_keys(PIECEWISE)` once. Keys are initialized correctly the
   first time — no re-initialization or stale FULL keys.

2. **`GMSWorker.initialize_from_config()`** — asserts the mode is correct after
   everything has run. Pure safety net. If the override worked, this never fires.
   If something changes upstream and the override is bypassed, this catches it.

```python
# GMSModelRunner
def _check_and_update_cudagraph_mode(self, attention_backends, kv_cache_groups):
    if is_shadow_mode():
        from vllm.config import CUDAGraphMode
        self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            CUDAGraphMode.PIECEWISE, self.uniform_decode_query_len
        )
        return
    return super()._check_and_update_cudagraph_mode(attention_backends, kv_cache_groups)

# GMSWorker
def initialize_from_config(self, kv_cache_config):
    super().initialize_from_config(kv_cache_config)
    if is_shadow_mode():
        from vllm.config import CUDAGraphMode
        mode = self.model_runner.compilation_config.cudagraph_mode
        if mode not in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE):
            raise RuntimeError(
                f"Shadow mode requires PIECEWISE or NONE cudagraph mode "
                f"after resolution, but got {mode.name}"
            )
```

## Call chain reference

```
Worker.initialize_from_config()                          [GMSWorker override]
  └─ model_runner.initialize_kv_cache()                  [GPUModelRunner]
       ├─ initialize_attn_backend()
       │    └─ _check_and_update_cudagraph_mode()        [GMSModelRunner override]
       │         └─ initialize_cudagraph_keys(PIECEWISE)
       ├─ initialize_metadata_builders()
       ├─ initialize_kv_cache_tensors()                  [GMSModelRunner override → {}]
       └─ register_kv_caches({})                         [monkey-patch guards this]
  └─ assert cudagraph mode is PIECEWISE or NONE          [GMSWorker safety net]
```
