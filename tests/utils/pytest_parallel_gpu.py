#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-parallel test runner (used by conftest.py, not invoked directly).

Runs pytest tests as independent subprocesses with VRAM-aware scheduling.
Each test gets CUDA_VISIBLE_DEVICES and _PROFILE_PYTEST_VRAM_FRAC_OVERRIDE
so the engine allocates only its declared VRAM budget.

Usage (always via pytest):
    pytest --max-vram-gib=6 -n auto -m "gpu_1 and vllm" tests/serve/
    pytest --max-vram-gib=6 -n 4 -sv -m "gpu_1 and vllm" tests/serve/

Flags:
    --max-vram-gib=N   Only run tests with profiled_vram_gib <= N
    -n N / -n auto     Run N tests concurrently (auto = GPU budget / smallest test)
    -s                 Stream subprocess output live with [wN] prefixes
    -v / -vv           Passed through to subprocesses for verbose test names

A 10-second cooldown between launches avoids the vLLM profiling race
(bug #10643). Tests that fail due to profiling race are retried up to 3 times.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pynvml


def _print(msg: str = "") -> None:
    """Print to stderr so pytest doesn't capture it."""
    print(msg, file=sys.stderr, flush=True)


_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tests.utils.vram_scheduler import (  # noqa: E402
    VRAM_SAFETY_FACTOR,
    auto_worker_count,
    detect_gpus,
    load_test_meta,
)

_JUNIT_DIR = os.path.join(tempfile.gettempdir(), "gpu_parallel_junit")
_JUNIT_COMBINED = os.path.join(_JUNIT_DIR, "combined.xml")


def _aggregate_junit_xml(junit_dir: str) -> str | None:
    """Merge per-test JUnit XML files into one combined testsuite."""
    import xml.etree.ElementTree as ET

    xmls = sorted(Path(junit_dir).glob("*.xml"))
    xmls = [x for x in xmls if x.name != "combined.xml"]
    if not xmls:
        return None

    total_tests = total_errors = total_failures = 0
    total_time = 0.0
    testcases = []

    for xml_path in xmls:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root = tree.getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            continue
        total_tests += int(suite.get("tests", 0))
        total_errors += int(suite.get("errors", 0))
        total_failures += int(suite.get("failures", 0))
        total_time += float(suite.get("time", 0))
        testcases.extend(suite.findall("testcase"))

    combined = ET.Element(
        "testsuite",
        {
            "name": "gpu-parallel",
            "tests": str(total_tests),
            "errors": str(total_errors),
            "failures": str(total_failures),
            "time": f"{total_time:.3f}",
        },
    )
    for tc in testcases:
        combined.append(tc)

    out = _JUNIT_COMBINED
    ET.ElementTree(combined).write(out, encoding="unicode", xml_declaration=True)
    return out


def _collect_tests(pytest_args: list[str], max_vram_gib: float) -> list[str]:
    """Run pytest --collect-only to get test IDs, filtered by --max-vram-gib."""
    _strip_flags = {"-v", "-vv", "-vvv", "--verbose", "-s", "--capture=no"}
    collect_args = [a for a in pytest_args if a not in _strip_flags]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        f"--max-vram-gib={max_vram_gib}",
        "--collect-only",
        "-q",
        *collect_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    test_ids = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if "::" in line and not line.startswith(" "):
            test_ids.append(line)
    return test_ids


def _get_gpu_used_gib(gpu_index: int = 0) -> float:
    """Query actual GPU memory used via pynvml."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return mem.used / (1024**3)
    except pynvml.NVMLError:
        return 0.0


def _short_name(test_id: str) -> str:
    return test_id.split("::", 1)[1] if "::" in test_id else test_id


_RETRYABLE_INIT_MARKERS = [
    "Error in memory profiling",  # vLLM profiling race assertion
    "Free memory on device",  # not enough free VRAM at startup
    "Engine core initialization failed",  # engine init crash
    "exited with code 0 while waiting for health check",  # engine started but died during init
    "exited with code -15 while waiting for health check",  # SIGTERM during init
    "exited with code -9 while waiting for health check",  # SIGKILL (OOM killer) during init
]
_MAX_RETRIES = 3


def _capture_output(pipe, captured: list[str], prefix: str | None = None) -> None:
    """Read all lines from a pipe into `captured`. Runs in a thread.

    If prefix is set, also prints each line live (-s mode).
    """
    for line in iter(pipe.readline, ""):
        line = line.rstrip("\n")
        if line:
            captured.append(line)
            if prefix is not None:
                _print(f"{prefix} {line}")
    pipe.close()


def run_parallel(
    test_ids: list[str],
    meta: dict[str, dict],
    max_vram_gib: float,
    num_slots: int,
    gpu_index: int = 0,
    extra_pytest_args: list[str] | None = None,
    stream: bool = False,
) -> int:
    """Run tests in parallel with VRAM-aware scheduling.

    Flags (mimic pytest semantics):
      -s       Stream subprocess output live with [wN] prefixes.
      -v/-vv   Passed through to subprocesses for verbose test names / diffs.
               No effect on the orchestrator's output.

    Without -s, output is buffered and printed after each test completes.
    Returns exit code: 0 if all pass, 1 if any fail.
    """
    gpus = detect_gpus()
    if not gpus:
        _print("ERROR: No GPUs detected")
        return 1

    gpu = gpus[gpu_index]
    gpu_total_gib = gpu["total_mib"] / 1024.0
    gpu_budget_gib = gpu_total_gib * VRAM_SAFETY_FACTOR

    tests = []
    for tid in test_ids:
        m = meta.get(tid, {})
        tests.append(
            {
                "id": tid,
                "name": _short_name(tid),
                "profiled_gib": m.get("profiled_vram_gib", max_vram_gib),
                "requested_gib": m.get("requested_vram_gib"),
                "timeout": m.get("timeout", 600),
            }
        )

    # Sort by timeout descending (longest first to minimize tail latency)
    tests.sort(key=lambda t: t["timeout"], reverse=True)

    # Identify tests in metadata that exceed the VRAM budget
    test_id_set = set(test_ids)
    skipped = []
    for nodeid, m in meta.items():
        if nodeid not in test_id_set:
            profiled = m.get("profiled_vram_gib")
            if profiled is not None and profiled > max_vram_gib:
                skipped.append((_short_name(nodeid), profiled))

    # Assign permanent worker IDs (w0, w1, ...) to each test
    for idx, test in enumerate(tests):
        test["w_id"] = idx

    os.makedirs(_JUNIT_DIR, exist_ok=True)

    # Print the test plan / worker assignment table
    _print(
        f"\nGPU parallel: {len(tests)} tests, {num_slots} concurrent slots, "
        f"GPU {gpu_index} ({gpu_total_gib:.0f} GiB, {gpu_budget_gib:.0f} GiB usable)"
    )
    max_name = max((len(t["name"]) for t in tests), default=30)
    _print()
    for test in tests:
        req = (
            f"{test['requested_gib']:>5.1f}"
            if test["requested_gib"] is not None
            else "    -"
        )
        label = f"[w{test['w_id']}] {test['name']}"
        _print(
            f"{label:<{max_name + 8}} "
            f"profiled={test['profiled_gib']:>5.1f}  req={req} GiB  "
            f"timeout={int(test['timeout'])}s"
        )
    if skipped:
        _print()
        _print(
            f"Skipped ({len(skipped)} -- profiled > max_vram_gib {max_vram_gib:.0f} GiB):"
        )
        for name, profiled in sorted(skipped, key=lambda x: x[1], reverse=True):
            _print(f"  {name}  (profiled {profiled:.1f} GiB)")
    _print()

    t0 = time.monotonic()
    pending = list(tests)
    # running keyed by w_id (permanent per-test), concurrency limited by slot count
    running: dict[int, dict] = {}  # w_id -> {proc, test, start_time, captured, retries}
    completed: list[dict] = []
    budget_used = 0.0
    next_status = t0 + 10
    # Cooldown after each launch to let the engine finish vLLM profiling
    # before the next engine starts (avoids profiling race, bug #10643).
    last_launch_time = 0.0
    launch_cooldown_s = 10.0

    def _launch_test(test: dict, env_base: dict) -> dict:
        """Build env, spawn subprocess, start output streamer thread."""
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        if test["requested_gib"] is not None:
            # Use profiled_gib (includes CUDA overhead) for the fraction so there's
            # enough headroom when other parallel processes also have CUDA contexts.
            fraction = min(test["profiled_gib"] / gpu_total_gib, 0.95)
            env["_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE"] = f"{fraction:.2f}"

        junit_path = os.path.join(_JUNIT_DIR, f"{test['name']}.xml")
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test["id"],
            "-x",
            "--tb=short",
            f"--timeout={int(test['timeout'])}",
            f"--junitxml={junit_path}",
        ]
        if extra_pytest_args:
            cmd.extend(extra_pytest_args)

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        captured: list[str] = []
        w_id = test["w_id"]
        stream_prefix = f"[w{w_id}]" if stream else None
        t = threading.Thread(
            target=_capture_output,
            args=(proc.stdout, captured, stream_prefix),
            daemon=True,
        )
        t.start()
        return {
            "proc": proc,
            "test": test,
            "start_time": time.monotonic(),
            "captured": captured,
        }

    env_base = os.environ.copy()

    while pending or running:
        now = time.monotonic()

        # Check for completed subprocesses
        for w_id in list(running.keys()):
            info = running[w_id]
            rc = info["proc"].poll()
            if rc is not None:
                duration = now - info["start_time"]
                passed = rc == 0
                test = info["test"]

                # Detect vLLM profiling race and retry silently
                if not passed and test.get("retries", 0) < _MAX_RETRIES:
                    hit_race = any(
                        marker in line
                        for line in info["captured"]
                        for marker in _RETRYABLE_INIT_MARKERS
                    )
                    if hit_race:
                        test["retries"] = test.get("retries", 0) + 1
                        _print(f"[w{w_id}] retrying ({test['retries']}/{_MAX_RETRIES})")
                        budget_used -= test["profiled_gib"]
                        del running[w_id]
                        pending.insert(0, test)
                        last_launch_time = 0.0  # skip cooldown for retry
                        continue

                # Dump buffered output on failure only (matches pytest behavior).
                # With -s, output was already streamed live.
                if not stream and not passed:
                    prefix = f"[w{w_id}]"
                    for line in info["captured"]:
                        _print(f"{prefix} {line}")

                status = "PASSED" if passed else "FAILED"
                # #region agent log
                _actual_at_done = _get_gpu_used_gib(gpu_index)
                _last_lines = info["captured"][-10:] if not passed else []
                _debug_done = json.dumps(
                    {
                        "sessionId": "8dc0f8",
                        "hypothesisId": "H1_H2_H3",
                        "location": "pytest_parallel_gpu.py:done",
                        "message": "done",
                        "data": {
                            "w_id": w_id,
                            "test": test["name"],
                            "passed": passed,
                            "rc": rc,
                            "duration_s": round(duration, 1),
                            "actual_gpu_used_gib": round(_actual_at_done, 1),
                            "budget_used_after": round(budget_used, 1),
                            "retries": test.get("retries", 0),
                            "error_tail": _last_lines,
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                with open("/workspace/.cursor/debug-8dc0f8.log", "a") as _df:
                    _df.write(_debug_done + "\n")
                # #endregion
                _print(f"[w{w_id}] {status} {test['name']} ({duration:.0f}s)")
                budget_used -= test["profiled_gib"]
                completed.append(
                    {
                        "test": test,
                        "duration": duration,
                        "passed": passed,
                    }
                )
                del running[w_id]

                # Print status immediately after completion
                elapsed = int(now - t0)
                actual_gib = _get_gpu_used_gib(gpu_index)
                still_running = ", ".join(
                    f"w{w}({int(now - running[w]['start_time'])}s)"
                    for w in sorted(running)
                )
                queued_str = ", ".join(f"w{t['w_id']}" for t in pending)
                parts = [
                    f"[elapsed {elapsed}s] GPU {gpu_index}: "
                    f"{actual_gib:.1f} GiB used / "
                    f"{gpu_total_gib - actual_gib:.1f} GiB free",
                ]
                if running:
                    parts.append(f"[running: {still_running}]")
                if pending:
                    parts.append(f"[queued: {queued_str}]")
                _print(" ".join(parts))
                next_status = now + 10  # reset periodic timer

        # Try to launch one pending test if cooldown has elapsed and slots available.
        # Gate on BOTH budget tracking (profiled) AND actual GPU free memory.
        can_launch = (now - last_launch_time) >= launch_cooldown_s
        if pending and can_launch and len(running) < num_slots:
            actual_used_gib = _get_gpu_used_gib(gpu_index)
            actual_free_gib = gpu_total_gib - actual_used_gib
            for i, test in enumerate(pending):
                budget_ok = budget_used + test["profiled_gib"] <= gpu_budget_gib
                actual_ok = actual_free_gib >= test["profiled_gib"]
                if budget_ok and actual_ok:
                    pending.pop(i)
                    w_id = test["w_id"]
                    budget_used += test["profiled_gib"]
                    info = _launch_test(test, env_base)
                    running[w_id] = info
                    free_gib = gpu_budget_gib - budget_used
                    retry_str = (
                        f" (retry {test.get('retries', 0)})"
                        if test.get("retries")
                        else ""
                    )
                    _print(
                        f"[w{w_id}] {test['name']} "
                        f"(profiled {test['profiled_gib']:.1f} GiB, "
                        f"budget {free_gib:.1f} free, "
                        f"actual {actual_free_gib:.1f} GiB free){retry_str}"
                    )
                    last_launch_time = now
                    break

        # Periodic status
        if now >= next_status and running:
            elapsed = int(now - t0)
            actual_gib = _get_gpu_used_gib(gpu_index)
            running_str = ", ".join(
                f"w{w}({int(now - running[w]['start_time'])}s)" for w in sorted(running)
            )
            queued_str = ", ".join(f"w{t['w_id']}" for t in pending)
            parts = [
                f"[elapsed {elapsed}s] GPU {gpu_index}: "
                f"{actual_gib:.1f} GiB used / "
                f"{gpu_total_gib - actual_gib:.1f} GiB free "
                f"[running: {running_str}]",
            ]
            if pending:
                parts.append(f"[queued: {queued_str}]")
            _print(" ".join(parts))
            next_status = now + 10

        if running:
            time.sleep(1.0)

    # Summary
    wall_time = time.monotonic() - t0
    sequential_time = sum(c["duration"] for c in completed)
    n_passed = sum(1 for c in completed if c["passed"])
    n_failed = sum(1 for c in completed if not c["passed"])

    # Sort by w_id for consistent display
    completed.sort(key=lambda c: c["test"]["w_id"])

    # Calculate column width from longest test name
    max_name = max((len(c["test"]["name"]) for c in completed), default=30)
    col_w = max_name + 8  # [wN] + padding

    _print()
    _print(f"{'=' * (col_w + 50)}")
    for c in completed:
        t = c["test"]
        status = "PASSED" if c["passed"] else "FAILED"
        retries = t.get("retries", 0)
        retry_str = f" ({retries} retries)" if retries else ""
        label = f"[w{t['w_id']}] {t['name']}"
        prof = f"{t['profiled_gib']:>5.1f}"
        req = (
            f"{t['requested_gib']:>5.1f}" if t["requested_gib"] is not None else "    -"
        )
        timeout = int(t["timeout"])
        _print(
            f"{label:<{col_w}} {c['duration']:>4.0f}s/{timeout:<3}s  "
            f"profiled={prof}  req={req} GiB  {status}{retry_str}"
        )
    _print(f"{'=' * (col_w + 50)}")

    if len(completed) > 1 and sequential_time > 0:
        speedup = sequential_time / wall_time
        saved = sequential_time - wall_time
        _print(
            f"{n_passed} passed, {n_failed} failed in {wall_time:.0f}s "
            f"(vs {sequential_time:.0f}s sequential) "
            f"-- {speedup:.1f}x speedup, saved {saved:.0f}s"
        )
    else:
        _print(f"{n_passed} passed, {n_failed} failed in {wall_time:.0f}s")

    # Aggregate JUnit XMLs into a single file
    combined = _aggregate_junit_xml(_JUNIT_DIR)
    if combined:
        _print(f"JUnit XML: {combined}")

    return 0 if n_failed == 0 else 1


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GPU tests in parallel with VRAM-aware scheduling.",
        usage="%(prog)s --max-vram-gib=N [-n SLOTS] [pytest-args...]",
    )
    parser.add_argument(
        "--max-vram-gib",
        type=float,
        required=True,
        help="Only run tests with profiled_vram_gib <= N.",
    )
    parser.add_argument(
        "-n",
        type=str,
        default="auto",
        help="Number of concurrent slots. 'auto' = gpu_usable / max_vram_gib.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index (default: 0).",
    )

    raw = sys.argv[1:]
    if "--" in raw:
        split = raw.index("--")
        args = parser.parse_args(raw[:split])
        pytest_args = raw[split + 1 :]
    else:
        args, pytest_args = parser.parse_known_args(raw)

    if not pytest_args:
        parser.error("No pytest arguments provided")

    is_stream = any(a in ("-s", "--capture=no") or "-s" in a for a in pytest_args)

    _print(f"Collecting tests with --max-vram-gib={args.max_vram_gib}...")
    test_ids = _collect_tests(pytest_args, args.max_vram_gib)
    if not test_ids:
        _print("No tests collected.")
        return 0

    meta = load_test_meta()

    if args.n == "auto":
        gpus = detect_gpus()
        if not gpus:
            _print("ERROR: No GPUs detected")
            return 1
        profiled_gibs = [
            meta.get(tid, {}).get("profiled_vram_gib", args.max_vram_gib)
            for tid in test_ids
        ]
        num_slots = auto_worker_count(gpus, args.max_vram_gib, profiled_gibs)
    else:
        num_slots = int(args.n)

    return run_parallel(
        test_ids=test_ids,
        meta=meta,
        max_vram_gib=args.max_vram_gib,
        num_slots=num_slots,
        gpu_index=args.gpu,
        stream=is_stream,
    )


if __name__ == "__main__":
    sys.exit(main())
