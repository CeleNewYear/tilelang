"""CPU-simulation entry points for the Ascend backend.

The implementation lives in :mod:`tilelang._ascend_sim` because it has to run
before ``tilelang`` finishes importing (and before torch/torch_npu are
imported), and therefore must not pull in this package's heavier imports. This
module simply re-exports it so ``tilelang.ascend.sim`` is the discoverable
public name.

Typical use::

    TILELANG_ASCEND_SIM=1 python my_kernel.py

with nothing else to change. From Python::

    from tilelang.ascend import sim

    if sim.active():
        print("running on the CANN CPU simulator")
    sim.synchronize()
"""

from __future__ import annotations

from tilelang._ascend_sim import (
    ENV_ENABLE,
    ENV_HARD_EXIT,
    ENV_LIB_DIR,
    ENV_SHIM,
    ENV_SOC,
    active,
    device_count,
    enable,
    hard_exit,
    maybe_enable,
    requested,
    sim_lib_dir,
    synchronize,
)

__all__ = [
    "ENV_ENABLE",
    "ENV_HARD_EXIT",
    "ENV_LIB_DIR",
    "ENV_SHIM",
    "ENV_SOC",
    "active",
    "device_count",
    "enable",
    "hard_exit",
    "maybe_enable",
    "requested",
    "sim_lib_dir",
    "synchronize",
]
