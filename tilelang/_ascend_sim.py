"""Optional CPU-simulation bootstrap for the Ascend backend.

CANN ships a cycle-accurate CPU simulator for Ascend 950 (``dav-3510``): the
*PEM model* in ``<CANN>/tools/simulator/dav_3510`` executes real AI-core
binaries produced by ``bisheng`` on the host CPU, with no NPU present. Running
TileLang kernels on it turns device code into something a CPU-only machine (or
CI runner) can execute and test.

How the simulator is reached
----------------------------
The simulator is not a separate API: CANN's ACL runtime resolves its
interposable ``cce::runtime`` symbols at load time, so whichever provider is
mapped first wins. Mapping ``libruntime_camodel.so`` and
``libnpu_drv_camodel.so`` before any other CANN library therefore redirects
*all* subsequent ACL calls -- TileLang's runtime module, ``torch_npu``'s device
allocations, synchronisation and memcpy -- to the simulated device.

Those libraries form one mutually recursive, eagerly bound dependency group
(``libstars.so`` <-> ``libffts_model.so``) without an ``$ORIGIN`` RPATH, so no
single member can be ``dlopen``ed first and the group can only be resolved as
part of one load request. This module ``dlopen``s
``libtilelang_ascend_sim.so`` -- the dependency-only shim built by
``-DTILELANG_ASCEND_SIM=ON`` (see ``src/ascend/sim/ascend_sim_shim.cc``) -- with
``RTLD_GLOBAL`` *before* torch/torch_npu are imported. The shim's ``DT_NEEDED``
entries and inherited ``DT_RPATH`` give the loader exactly that one request.

Environment variables
---------------------
``TILELANG_ASCEND_SIM``
    Master switch. Unset (default) means "do nothing": a normal NPU build keeps
    its full host-side behaviour and pays a single ``os.environ`` lookup at
    import. Any of ``1/true/on/yes`` enables simulation; the value may also be
    a path to the simulator ``lib`` directory (or to a directory containing
    one), which is equivalent to setting ``TILELANG_ASCEND_SIM_LIB_DIR``.
``TILELANG_ASCEND_SIM_LIB_DIR``
    Explicit path to the directory holding ``libruntime_camodel.so``.
    Defaults to ``<CANN>/tools/simulator/<soc dir>/lib``, resolved from
    ``ASCEND_HOME_PATH`` / ``ASCEND_TOOLKIT_HOME`` / the ``bisheng`` location.
``TILELANG_ASCEND_SIM_SHIM``
    Explicit path to ``libtilelang_ascend_sim.so`` when it lives outside the
    package library directory (for example a custom build tree).
``TILELANG_ASCEND_SIM_SOC``
    SoC name exported as ``CAMODEL_SOC_VERSION`` for the simulator, e.g.
    ``Ascend950PR_9599``. Optional: the model reports its own version when the
    variable is absent.
``TILELANG_ASCEND_SIM_HARD_EXIT``
    Defaults to on. The simulator's teardown is not safe to run during Python
    interpreter finalisation (it aborts with ``std::bad_function_call`` or
    faults), so a small ``atexit`` hook flushes the streams and terminates the
    process with ``os._exit`` while the exit status is still meaningful.
    Set to ``0`` to keep the interpreter's own shutdown path; the process then
    usually reports ``SIGSEGV``/``SIGABRT`` after all work has finished.

Notes
-----
* Enable this **before** torch or torch_npu is imported. ``import tilelang``
  does that automatically, which is why the hook lives at the top of
  ``tilelang/__init__.py``. A late enablement is diagnosed at runtime.
* The computation really runs on the CPU: expect roughly a millionth of device
  throughput, so keep simulation workloads small (correctness, not performance).
* This module deliberately imports nothing beyond the standard library: it runs
  while ``tilelang`` itself is still initialising.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import threading

ENV_ENABLE = "TILELANG_ASCEND_SIM"
ENV_LIB_DIR = "TILELANG_ASCEND_SIM_LIB_DIR"
ENV_SHIM = "TILELANG_ASCEND_SIM_SHIM"
ENV_SOC = "TILELANG_ASCEND_SIM_SOC"
ENV_HARD_EXIT = "TILELANG_ASCEND_SIM_HARD_EXIT"

_TRUTHY = frozenset({"1", "true", "on", "yes"})
_FALSY = frozenset({"0", "false", "off", "no"})

_SIM_LIBRARIES = ("libruntime_camodel.so", "libnpu_drv_camodel.so")
_SIM_SHIM_NAME = "libtilelang_ascend_sim.so"

# SoC name prefix -> simulator directory under ``<CANN>/tools/simulator``.
# CANN ships per-architecture directories (``dav_3510`` for Ascend 950,
# ``dav_2201`` for Ascend 910B, ...) plus SoC-name symlinks into them.
_SOC_TO_SIM_DIR = (
    ("Ascend950", "dav_3510"),
    ("Ascend960", "dav_3510"),
    ("Ascend910B", "dav_2201"),
    ("Ascend910_93", "dav_2201"),
    ("Ascend910A", "dav_1001"),
    ("Ascend310P", "dav_2002"),
    ("Ascend310", "dav_3002"),
)
_FALLBACK_SIM_DIR = "dav_3510"

_LOCK = threading.Lock()
_STATE = {
    "active": False,
    "lib_dir": None,
    "shim": None,
    "exit_hook": False,
}


def _warn(message: str, *, error: bool = False) -> None:
    """Report a simulation bootstrap problem without importing logging."""
    prefix = "TileLang Ascend simulation error: " if error else "TileLang Ascend simulation: "
    print(prefix + message, file=sys.stderr, flush=True)


def _flag(value: str | None) -> bool | None:
    """Parse a boolean-ish environment value, returning None when unset."""
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    return None


def requested() -> bool:
    """Return whether simulation was requested via the environment."""
    value = os.environ.get(ENV_ENABLE)
    if value is None or not value.strip():
        return False
    # Any non-false-like value enables simulation; a path value also selects the
    # simulator library directory (see ``sim_lib_dir``).
    return _flag(value) is not False


def active() -> bool:
    """Return whether the simulator runtime is mapped into this process."""
    return bool(_STATE["active"])


def sim_lib_dir() -> str | None:
    """Return the simulator library directory, or None when it is not found."""
    explicit = os.environ.get(ENV_LIB_DIR)
    value = os.environ.get(ENV_ENABLE)
    if value and _flag(value) is None:
        # TILELANG_ASCEND_SIM itself pointed at a directory.
        explicit = explicit or value
    if explicit:
        for candidate in (explicit, os.path.join(explicit, "lib")):
            if os.path.isfile(os.path.join(candidate, _SIM_LIBRARIES[0])):
                return candidate

    sim_dir = _sim_dir_for_soc(_soc_version())
    for root in _cann_roots():
        for sub in (
            os.path.join("tools", "simulator", sim_dir, "lib"),
            os.path.join("x86_64-linux", "simulator", sim_dir, "lib"),
            os.path.join("simulator", sim_dir, "lib"),
        ):
            candidate = os.path.join(root, sub)
            if os.path.isfile(os.path.join(candidate, _SIM_LIBRARIES[0])):
                return candidate
    return None


def _soc_version() -> str:
    value = os.environ.get(ENV_SOC) or os.environ.get("CAMODEL_SOC_VERSION")
    return value.strip() if value and value.strip() else "Ascend950PR_9599"


def _sim_dir_for_soc(soc_version: str) -> str:
    for prefix, sim_dir in _SOC_TO_SIM_DIR:
        if soc_version.startswith(prefix):
            return sim_dir
    return _FALLBACK_SIM_DIR


def _cann_roots() -> list[str]:
    roots: list[str] = []
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME"):
        value = os.environ.get(key)
        if value:
            roots.append(value)
    bisheng = _which("bisheng")
    if bisheng:
        roots.append(os.path.dirname(os.path.dirname(os.path.abspath(bisheng))))
    roots.append("/usr/local/Ascend/ascend-toolkit/latest")
    seen: set[str] = set()
    return [root for root in roots if not (root in seen or seen.add(root))]


def _which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _package_lib_dirs() -> list[str]:
    """Return the library directories that may hold the simulation shim."""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(package_dir)
    return [
        os.path.join(package_dir, "lib"),  # wheel / installed layout
        os.path.join(repo_root, "build", "lib"),  # source checkout layout
    ]


def _shim_path() -> str | None:
    explicit = os.environ.get(ENV_SHIM)
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for directory in _package_lib_dirs():
        candidate = os.path.join(directory, _SIM_SHIM_NAME)
        if os.path.isfile(candidate):
            return candidate
    return None


def _camodel_mapped() -> bool:
    """Return whether the simulator runtime is already mapped in this process."""
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as maps:
            for line in maps:
                if "libruntime_camodel.so" in line:
                    return True
    except OSError:
        # Not Linux, or /proc unavailable: fall back to the recorded state.
        return bool(_STATE["active"] or os.environ.get("IS_SIMULATOR_ENV"))
    return False


def _load_shim(lib_dir: str) -> bool:
    shim = _shim_path()
    if shim is None:
        return False
    try:
        _STATE["shim"] = ctypes.CDLL(shim, mode=os.RTLD_GLOBAL | os.RTLD_NOW)
    except OSError as exc:
        _warn(
            f"could not load {shim}: {exc}. Rebuild with -DTILELANG_ASCEND_SIM=ON "
            f"against the CANN installation you are running, or point "
            f"{ENV_SHIM} at a matching libtilelang_ascend_sim.so.",
        )
        return False
    return True


def _load_members_directly(lib_dir: str) -> bool:
    """Best-effort fallback: map the simulator libraries one by one.

    This only succeeds when the loader can resolve the group through
    ``LD_LIBRARY_PATH`` (for example a process started with the simulator
    preloaded by an external harness). It is kept as a fallback so a build
    without the shim is still usable.
    """
    mode = os.RTLD_GLOBAL | os.RTLD_LAZY
    for name in _SIM_LIBRARIES:
        try:
            ctypes.CDLL(os.path.join(lib_dir, name), mode=mode)
        except OSError:
            continue
    return _camodel_mapped()


def _warn_if_too_late() -> None:
    # torch on its own only loads its device backends lazily; torch_npu is what
    # maps libascendcl and would bind the ACL runtime to the real device.
    if sys.modules.get("torch_npu") is None:
        return
    _warn(
        "torch_npu was imported before simulation was enabled, so its ACL "
        "runtime may already be bound to the device runtime. Set "
        f"{ENV_ENABLE} in the environment (or import tilelang before "
        "torch_npu) so the simulator is mapped first.",
    )


def _install_exit_hook() -> None:
    """Terminate cleanly under the simulator's non-reentrant teardown."""
    if _STATE["exit_hook"]:
        return

    import atexit

    status = {"code": 0, "failed": False}
    previous_excepthook = sys.excepthook
    previous_exit = sys.exit

    def _excepthook(exc_type, exc, traceback):
        status["failed"] = True
        previous_excepthook(exc_type, exc, traceback)

    def _exit(code=0):
        status["code"] = code if isinstance(code, int) else 1
        previous_exit(code)

    def _cleanup():
        with contextlib.suppress(Exception):
            synchronize()
        with contextlib.suppress(Exception):
            sys.stdout.flush()
            sys.stderr.flush()
        code = status["code"] or (1 if status["failed"] else 0)
        os._exit(code)

    sys.excepthook = _excepthook
    sys.exit = _exit
    atexit.register(_cleanup)
    _STATE["exit_hook"] = True


def _sim_log_dir() -> str:
    """Return a private directory for the simulator's log and dump files.

    The model writes per-core instruction/CCU dumps and ``profile_*.toml``
    files into its log directory; left unset it uses the current working
    directory, which would scatter files through the user's tree.
    """
    cache_root = os.environ.get("TILELANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".tilelang", "cache")
    path = os.path.join(cache_root, "ascend_sim", "log")
    os.makedirs(path, exist_ok=True)
    return path


def _activate(lib_dir: str) -> None:
    os.environ.setdefault("IS_SIMULATOR_ENV", "true")
    os.environ.setdefault("CAMODEL_SOC_VERSION", _soc_version())
    try:
        log_dir = _sim_log_dir()
    except OSError:
        log_dir = None
    if log_dir is not None:
        os.environ.setdefault("CAMODEL_LOG_PATH", log_dir)
        os.environ.setdefault("STARS_LOG_PATH", log_dir)
    already_imported = sys.modules.get("torch_npu") is not None
    _STATE["active"] = True
    _STATE["lib_dir"] = lib_dir
    # Bind torch_npu now, while the simulator is the only mapped CANN runtime.
    # torch_npu caches its device state on first import, so importing it after
    # an unmapped (or real) runtime loads leaves it permanently device-less.
    _bind_torch_npu(already_imported)
    # pytest decides the process exit status itself and raises SystemExit
    # directly, which this hook cannot observe; exiting at interpreter teardown
    # would turn a failing session into a passing exit code.  The Ascend test
    # suite's conftest terminates with the session status instead.
    if _flag(os.environ.get(ENV_HARD_EXIT)) is not False and not _under_pytest():
        _install_exit_hook()


def _under_pytest() -> bool:
    """Return whether this process is a pytest run."""
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


def _bind_torch_npu(already_imported: bool) -> None:
    """Import torch_npu so that it initialises against the simulator."""
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the installation
        _warn(f"could not import torch_npu for the simulated device: {exc}")
        return
    if already_imported and not torch.npu.is_available():
        # The ACL runtime was bound before the simulator was mapped; importing
        # torch_npu again is a no-op, so report the required ordering.
        _warn(
            "torch_npu was imported before the simulator was mapped and reports "
            "no device. Import tilelang before torch/torch_npu, or set "
            "TORCH_DEVICE_BACKEND_AUTOLOAD=0 so torch does not import torch_npu "
            "early.",
        )


def enable(*, quiet: bool = False) -> bool:
    """Map the CANN simulator runtime into this process.

    Idempotent. Must run before torch/torch_npu are imported to take effect;
    returns True when the simulator is (or already was) mapped.

    Raises
    ------
    RuntimeError
        When simulation was requested but no simulator runtime could be mapped.
        Continuing silently would run the kernel against a device runtime that
        is absent on this machine, producing a much less useful failure later.
    """
    with _LOCK:
        if _STATE["active"]:
            return True

        lib_dir = sim_lib_dir()
        if lib_dir is None:
            raise RuntimeError(
                f"{ENV_ENABLE} is set but no CANN simulator runtime was found. "
                f"Looked for {_SIM_LIBRARIES[0]} under "
                f"<CANN>/tools/simulator/{_sim_dir_for_soc(_soc_version())}/lib "
                f"for SoC {_soc_version()!r}. Source the CANN set_env.sh, or set "
                f"{ENV_LIB_DIR} to the directory holding the simulator libraries."
            )

        if _camodel_mapped():
            _activate(lib_dir)
            return True

        _warn_if_too_late()

        loaded = _load_shim(lib_dir) or _load_members_directly(lib_dir)
        if not loaded and not _camodel_mapped():
            raise RuntimeError(
                f"{ENV_ENABLE} is set but the Ascend simulator runtime could not "
                f"be mapped from {lib_dir}. Build TileLang with "
                f"-DTILELANG_ASCEND_SIM=ON so libtilelang_ascend_sim.so is "
                f"produced, or start the process with "
                f"LD_PRELOAD='{lib_dir}/libruntime_camodel.so "
                f"{lib_dir}/libnpu_drv_camodel.so' and "
                f"LD_LIBRARY_PATH='{lib_dir}:$LD_LIBRARY_PATH'."
            )

        _activate(lib_dir)
        if not quiet:
            _warn(f"running on the CANN CPU simulator at {lib_dir}")
        return True


def maybe_enable() -> bool:
    """Enable simulation when requested; a no-op otherwise.

    Called from ``tilelang/__init__.py`` before torch is imported, so it must
    stay free of heavyweight imports.
    """
    if not requested():
        return False
    return enable()


def synchronize() -> None:
    """Wait for simulated device work to finish (no-op without a device)."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        return
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def device_count() -> int:
    """Return the number of simulated devices available (0 or 1)."""
    try:
        import torch
    except Exception:  # pragma: no cover
        return 0
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return 0
    return int(torch.npu.device_count())


def hard_exit(code: int = 0) -> None:
    """Synchronise, flush and leave the process without simulator teardown."""
    with contextlib.suppress(Exception):
        synchronize()
    with contextlib.suppress(Exception):
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)


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
