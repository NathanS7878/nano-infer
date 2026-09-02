"""Custom CUDA kernels, JIT-compiled on first use.

torch.utils.cpp_extension.load compiles the .cu sources with nvcc and caches the
result, so the cost is paid once per machine rather than on every run.

Build environment (see HARDWARE.md):
  - nvcc 12.4 from conda-forge, matching the torch cu124 wheel. Installed via
    conda deliberately: the official CUDA installer bundles a display driver and
    would have downgraded this machine's newer driver for no benefit.
  - MSVC 14.44 (VS 2022 Build Tools) as the host compiler. CUDA 12.4 predates
    this MSVC and refuses it by default, so -allow-unsupported-compiler is passed.
  - -gencode for sm_86 only (RTX 3070, Ampere) — no point building for other
    architectures on a single-GPU project.

--use_fast_math is deliberately NOT set: it would change numerics and the whole
point is bit-level comparability against the PyTorch reference.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_KERNEL_DIR = Path(__file__).resolve().parent
_module = None

# Ampere sm_86 — the RTX 3070 this project targets.
_ARCH = "compute_86,code=sm_86"

_NVCC_FLAGS = [
    "-O3",
    "--expt-relaxed-constexpr",
    f"-gencode=arch={_ARCH}",
    "-allow-unsupported-compiler",   # MSVC 14.44 is newer than CUDA 12.4 expects
]

_CXX_FLAGS = ["/O2"] if sys.platform == "win32" else ["-O3"]


def _ensure_build_env() -> None:
    """Point the build at the conda CUDA toolkit, ninja, and the MSVC host compiler."""
    # ninja ships into the env's Scripts/bin dir, which pip warns is not on PATH.
    # torch shells out to `ninja --version`, so it has to be findable by name.
    prefix = Path(sys.executable).parent
    for d in (prefix / "Scripts", prefix / "bin", prefix):
        if d.exists() and str(d) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

    if not os.environ.get("CUDA_HOME"):
        # conda-forge installs nvcc under the env prefix.
        for cand in (prefix, prefix / "Library"):
            if (cand / "bin" / "nvcc.exe").exists() or (cand / "bin" / "nvcc").exists():
                os.environ["CUDA_HOME"] = str(cand)
                break

    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        nvcc_dir = str(Path(cuda_home) / "bin")
        if nvcc_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nvcc_dir + os.pathsep + os.environ.get("PATH", "")

    if sys.platform == "win32":
        _add_msvc_to_path()


def _add_msvc_to_path() -> None:
    """Put cl.exe on PATH. nvcc shells out to the host compiler by name."""
    from shutil import which
    if which("cl"):
        return
    roots = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
    ]
    for root in roots:
        base = root / "Microsoft Visual Studio" / "2022"
        if not base.exists():
            continue
        for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
            msvc = base / edition / "VC" / "Tools" / "MSVC"
            if not msvc.exists():
                continue
            versions = sorted(msvc.iterdir(), reverse=True)
            for v in versions:
                cl_dir = v / "bin" / "Hostx64" / "x64"
                if (cl_dir / "cl.exe").exists():
                    os.environ["PATH"] = str(cl_dir) + os.pathsep + os.environ["PATH"]
                    return


def _cuda_link_flags() -> list:
    """Locate the CUDA import libraries.

    torch's Windows build assumes the NVIDIA installer layout ($CUDA_HOME/lib/x64).
    conda-forge instead puts cudart.lib and friends directly in Library/lib, so
    the default -LIBPATH finds nothing and the link fails with LNK1181. Point the
    linker at whichever directory actually holds cudart.lib.
    """
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        return []
    root = Path(cuda_home)
    for cand in (root / "lib" / "x64", root / "lib64", root / "lib"):
        if (cand / "cudart.lib").exists() or (cand / "libcudart.so").exists():
            if sys.platform == "win32":
                return [f"/LIBPATH:{cand}"]
            return [f"-L{cand}"]
    return []


def load(verbose: bool = False):
    """Compile (once) and return the kernel module."""
    global _module
    if _module is not None:
        return _module

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; kernels require a GPU")

    _ensure_build_env()
    from torch.utils.cpp_extension import load as _load

    _module = _load(
        name="nano_infer_kernels",
        sources=[
            str(_KERNEL_DIR / "bindings.cpp"),
            str(_KERNEL_DIR / "rmsnorm.cu"),
            str(_KERNEL_DIR / "swiglu.cu"),
            str(_KERNEL_DIR / "rope.cu"),
        ],
        extra_cflags=_CXX_FLAGS,
        extra_cuda_cflags=_NVCC_FLAGS,
        extra_ldflags=_cuda_link_flags(),
        verbose=verbose,
    )
    return _module


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused RMSNorm. Drop-in replacement for nano_infer.model.rms_norm."""
    return load().rmsnorm_forward(x, weight, float(eps))


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU. Drop-in replacement for `F.silu(gate) * up`."""
    return load().swiglu_forward(gate, up)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Fused RoPE. Drop-in for `x * cos + rotate_half(x) * sin`.

    x        : [batch, heads, n, head_dim]
    cos, sin : [n, head_dim] (shared positions) or [batch, n, head_dim]
               (per-sequence positions, as continuous batching needs)
    """
    return load().rope_forward(x, cos, sin)
