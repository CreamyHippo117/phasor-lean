# PyInstaller recipe for the phasor desktop app (see packaging/README.md).
#   uv sync --extra gl --extra encode --extra gui --extra mcp --group package
#   uv run pyinstaller packaging/phasor.spec --noconfirm
# Output: dist/phasor/ (a folder app; on macOS also dist/phasor.app).
# Build on each platform for that platform: PyInstaller does not cross-compile.
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent
ICON = ROOT / "phasor" / "gui" / "assets" / ("icon.ico" if sys.platform == "win32" else "icon.png")

# Entry points (CLI commands, editor tabs) are found through package metadata.
datas = copy_metadata("phasor")
datas += collect_data_files("phasor", include_py_files=False)  # themes, looks, icons
datas += collect_data_files("matplotlib", subdir="mpl-data")  # mathtext fonts
datas += [(str(ROOT / "docs" / "ai"), "phasor/_docs/docs/ai"),
          (str(ROOT / "examples"), "examples"),
          (str(ROOT / "kit"), "phasor/_docs/kit"),  # the AI kit (phasor kit)
          (str(ROOT / "skills"), "phasor/_docs/skills"),
          (str(ROOT / "STYLE.md"), "phasor/_docs")]
hidden = collect_submodules("phasor") + ["moderngl", "glcontext", "av", "kiwisolver"]
binaries = []
excludes = ["cupy", "tkinter", "IPython", "ipywidgets", "notebook", "pytest"]
# PHASOR_PACKAGE_GPU=1 bundles CuPy and the CUDA libraries it needs (NVIDIA machines; about
# 340 MB more). The app still runs on a machine without a card: the kernels fall back to NumPy.
# phasor's kernels need only NVRTC and the CUDA runtime: cuBLAS, cuSOLVER, cuSPARSE, cuFFT,
# cuRAND and nvJitLink (2 GB) are left out; checked on an RTX 4070 Ti, 2026-11-20.
if os.environ.get("PHASOR_PACKAGE_GPU"):
    excludes.remove("cupy")
    hidden += ["graphlib"]  # CuPy imports it out of PyInstaller's sight (Windows, 2026-11-20)
    for package in ("cupy", "cupy_backends", "fastrlock", "nvidia.cuda_runtime",
                    "nvidia.cuda_nvrtc"):
        try:
            d, b, h = collect_all(package)
        except Exception:  # nvidia.* wheels are absent with older CuPy
            continue
        datas += d
        binaries += b
        hidden += h

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="phasor", icon=str(ICON),
    console=False,  # no terminal window behind the app
)
coll = COLLECT(exe, a.binaries, a.datas, name="phasor")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="phasor.app", icon=str(ICON), bundle_identifier="org.phasor.app")
