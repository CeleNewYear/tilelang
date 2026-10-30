"""Gate the Ascend suite on a build that includes the Ascend backend.

The Ascend sources are gated behind USE_ASCEND, so a CUDA-only build has no
Ascend target, no tl.ascend_*/tl.simd.* ops and no Ascend pass-config options.
Collecting these tests there produces a wall of unrelated failures instead of one
clear statement that the backend is not in the library.

The gate has to act at collection time rather than through a per-item mark.
Several modules do backend work at import scope -- creating an "ascend" Target,
looking up tl.simd.*/tl.ascend_* ops, importing torch_npu -- which raises while
pytest is still importing them, before any mark could apply. Ignoring the
directory keeps those imports from happening at all.

Per-test marks still apply on top of this: requires_ascend also carries the
run-time (device present) check, which this hook deliberately does not make, so
source-only lowering tests keep running on a host without an NPU.
"""

from __future__ import annotations

import os

import tilelang.testing

SKIP_REASON = "Ascend backend is not built into this library (configure with USE_ASCEND=ON)"

ASCEND_SUITE_GATED = not tilelang.testing.ascend_backend_compiled()

# Session exit status captured for the simulation teardown hook below.
_SESSION_EXITSTATUS = {"value": 0}

if ASCEND_SUITE_GATED:
    collect_ignore_glob = ["*"]


def pytest_collection_modifyitems(config, items):
    if ASCEND_SUITE_GATED:
        # testing/conftest.py turns this into a one-line note instead of its
        # "No tests were collected" error, which is meant for misconfigured runs.
        config._ascend_suite_gated = True


def pytest_sessionfinish(session, exitstatus):
    _SESSION_EXITSTATUS["value"] = int(exitstatus)


def pytest_unconfigure(config):
    """Terminate a simulated run instead of letting the model tear down.

    The CANN camodel runtime cannot run its teardown during CPython interpreter
    finalization: it aborts with ``std::bad_function_call`` or faults after all
    work has completed, which would report a passing run as SIGABRT/SIGSEGV.
    ``pytest_unconfigure`` is the last hook, after the summary is printed, so
    exiting here keeps both the output and pytest's own exit status.
    """
    if not _simulation_active():
        return
    if not _hard_exit_enabled():
        return
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_SESSION_EXITSTATUS["value"])


def _simulation_active() -> bool:
    try:
        from tilelang.ascend import sim
    except Exception:  # pragma: no cover - backend not importable
        return False
    return sim.active()


def _hard_exit_enabled() -> bool:
    from tilelang.ascend import sim

    value = os.environ.get(sim.ENV_HARD_EXIT, "").strip().lower()
    return value not in ("0", "false", "off", "no")
