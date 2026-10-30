"""Verify TileLang Ascend kernels execute on the CANN CPU simulator.

The simulator is opt-in: it is reached by mapping CANN's ``camodel`` runtime
before any other CANN library, which ``import tilelang`` does when
``TILELANG_ASCEND_SIM`` is set (see ``tilelang/_ascend_sim.py``). Without that
variable the whole feature is inert, so these tests skip rather than fail on a
normal NPU host.

Keep the workloads tiny: the simulator models AI-core execution on the host CPU,
so a few thousand elements already take seconds.
"""

from __future__ import annotations

import os

import pytest
import torch

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend import sim

pytest.importorskip("torch_npu")

ELEMENTS = 2048

requires_simulator = pytest.mark.skipif(
    not sim.active(),
    reason="set TILELANG_ASCEND_SIM=1 on a host with the CANN simulator to run CPU-simulation tests",
)


def build_vector_add():
    """Build a one-block SIMD vector add that exercises GM->UB->GM traffic.

    Auto-scheduling stays enabled: it is what inserts the cross-pipeline
    synchronization between the vector writes and the UB->GM copy, and without
    it the kernel races on a simulated device.
    """

    @tilelang.jit(target="ascend", out_idx=-1)
    def kernel():
        @T.prim_func
        def vector_add(
            a: T.Tensor((ELEMENTS,), T.float32),
            b: T.Tensor((ELEMENTS,), T.float32),
            c: T.Tensor((ELEMENTS,), T.float32),
        ):
            with T.Kernel(1) as _:
                a_ub = T.alloc_shared((ELEMENTS,), T.float32)
                b_ub = T.alloc_shared((ELEMENTS,), T.float32)
                c_ub = T.alloc_shared((ELEMENTS,), T.float32)
                T.copy(a[0:ELEMENTS], a_ub)
                T.copy(b[0:ELEMENTS], b_ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for index in range(ELEMENTS // 64):
                        a_vec = T.simd.vld(a_ub[index * 64])
                        b_vec = T.simd.vld(b_ub[index * 64])
                        sum_vec = T.simd.vadd(a_vec, b_vec, mask)
                        T.simd.vsts(c_ub[index * 64], sum_vec, mask)
                T.copy(c_ub, c[0:ELEMENTS])

        return vector_add

    return kernel()


def test_simulation_state_matches_environment() -> None:
    """The bootstrap state must reflect whether simulation was requested."""
    if sim.requested():
        assert sim.active(), "TILELANG_ASCEND_SIM was set but the simulator runtime is not mapped"
        lib_dir = sim.sim_lib_dir()
        assert lib_dir is not None
        assert os.path.isfile(os.path.join(lib_dir, "libruntime_camodel.so"))
        assert sim.device_count() >= 1
    else:
        assert not sim.active()


@requires_simulator
@tilelang.testing.requires_ascend
def test_vector_add_runs_on_simulator() -> None:
    """A compiled kernel must produce device-equivalent results on the CPU model."""
    kernel = build_vector_add()

    a = torch.randn(ELEMENTS)
    b = torch.randn(ELEMENTS)
    result = kernel(a.npu(), b.npu())
    sim.synchronize()

    torch.testing.assert_close(result.cpu(), a + b, rtol=0, atol=0)


@requires_simulator
@tilelang.testing.requires_ascend
def test_device_allocation_round_trips() -> None:
    """Device memory in the simulator must behave like NPU memory for Torch."""
    host = torch.arange(64, dtype=torch.float32)
    device = host.npu()
    torch.testing.assert_close(device.cpu(), host)
    torch.testing.assert_close((device * 2).cpu(), host * 2)
    sim.synchronize()
