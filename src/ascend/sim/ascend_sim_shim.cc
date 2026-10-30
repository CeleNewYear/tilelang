/**
 * \file ascend_sim_shim.cc
 * \brief Dependency-only shim that exposes the CANN CPU simulator runtime.
 *
 * The Ascend 950 CPU simulator (CANN's camodel, also printed as "PEM MODEL")
 * executes real dav-3510 AI-core binaries on the host CPU. To reach it, the
 * simulator runtime (libruntime_camodel.so + libnpu_drv_camodel.so) must be
 * mapped into the process before any CANN library is loaded, so that the ACL
 * runtime binds its interposable cce::runtime symbols to the simulator instead
 * of the real device runtime.
 *
 * A Python process cannot dlopen those libraries directly: the simulator
 * runtime is one mutually recursive, eagerly bound dependency group
 * (libstars.so <-> libffts_model.so), so no member can be loaded first, and the
 * group carries no $ORIGIN RPATH that would let the loader find its own
 * dependencies. The dynamic loader resolves such a group only when it is part
 * of a single load request.
 *
 * This translation unit therefore carries no code of interest: the shared
 * object built from it exists solely so the loader receives one request that
 * lists the whole simulator group, with an inherited (old-style DT_RPATH)
 * search path back to the CANN simulator directory. tilelang/_ascend_sim.py
 * dlopens the resulting libtilelang_ascend_sim.so with RTLD_GLOBAL and
 * RTLD_NOW before importing torch.
 *
 * The library is built only when -DTILELANG_ASCEND_SIM=ON is configured, so a
 * default build carries no reference to CANN's simulator at all.
 */

namespace {

// Keeps the shared object non-empty for toolchains and linkers that reject (or
// silently drop) a translation unit without any definitions.
int tilelang_ascend_sim_anchor = 0;

}  // namespace

extern "C" int tilelang_ascend_sim_shim_anchor(void) {
  return tilelang_ascend_sim_anchor;
}
