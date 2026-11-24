"""``phasor doctor``: prove what this machine can do and assign a hardware tier.

Each probe *exercises* the capability instead of trusting a registry: it compiles
and runs a kernel, renders a triangle, encodes frames. Listing is not proof; for
example PyAV wheels list ``h264_nvenc`` on machines with no NVIDIA GPU at all.

Every GPU/GL/video import is local to its probe, so this module imports fine on a
tier C machine and each probe fails on its own.
"""

from __future__ import annotations

import io
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from phasor.core.proc import NO_WINDOW

# Encoder probe order.
# H.264 before HEVC: it plays everywhere.
PYAV_ENCODERS = ("h264_nvenc", "hevc_nvenc", "libx264")
FFMPEG_ENCODERS = ("h264_nvenc", "hevc_nvenc", "libx264")


@dataclass
class Check:
    name: str
    ok: bool | None  # None = skipped because a prerequisite failed or is missing
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class DoctorReport:
    checks: list[Check]
    tier: str
    encoder: str | None

    def get(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "encoder": self.encoder,
            "checks": [asdict(c) for c in self.checks],
        }


def _guard(name: str, fn: Callable[[], Check]) -> Check:
    try:
        return fn()
    except Exception as e:  # a probe must never crash the doctor
        return Check(name, False, f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ python


def check_python() -> Check:
    v = sys.version_info
    ok = v >= (3, 11)
    return Check(
        "python",
        ok,
        f"Python {platform.python_version()} on {platform.platform()}"
        + ("" if ok else " (need 3.11+)"),
        {"version": platform.python_version(), "platform": platform.platform()},
    )


# ------------------------------------------------------------------ CUDA


def check_cuda() -> Check:
    try:
        import cupy
    except ImportError:
        return Check("cuda", None, "CuPy not installed (pip install phasor[gpu])")
    rt = cupy.cuda.runtime
    n = rt.getDeviceCount()
    if n < 1:
        return Check("cuda", False, "CuPy installed but no CUDA device found")
    props = rt.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    free, total = rt.memGetInfo()
    data = {
        "device": name,
        "device_count": n,
        "compute_capability": f"{props['major']}.{props['minor']}",
        "vram_total_mb": total // 2**20,
        "vram_free_mb": free // 2**20,
        "driver_version": rt.driverGetVersion(),
        "runtime_version": rt.runtimeGetVersion(),
        "cupy_version": cupy.__version__,
    }
    return Check(
        "cuda",
        True,
        f"{name}, {data['vram_free_mb']}/{data['vram_total_mb']} MB free, "
        f"CC {data['compute_capability']}, CuPy {cupy.__version__}",
        data,
    )


def check_nvrtc() -> Check:
    import cupy

    major, minor = cupy.cuda.nvrtc.getVersion()
    return Check("nvrtc", True, f"NVRTC {major}.{minor}", {"version": f"{major}.{minor}"})


_ADD_KERNEL = r"""
extern "C" __global__ void phasor_doctor_axpb(const float* x, float* y, float a, float b, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b;
}
"""


def check_rawkernel() -> Check:
    import cupy
    import numpy as np

    n = 4096
    kernel = cupy.RawKernel(_ADD_KERNEL, "phasor_doctor_axpb")
    x = cupy.arange(n, dtype=cupy.float32)
    y = cupy.empty_like(x)
    kernel(((n + 255) // 256,), (256,), (x, y, np.float32(2.0), np.float32(1.0), np.int32(n)))
    got = cupy.asnumpy(y)
    want = 2.0 * np.arange(n, dtype=np.float32) + 1.0
    ok = bool(np.allclose(got, want))
    cupy.get_default_memory_pool().free_all_blocks()
    return Check("rawkernel", ok, "compiled and ran via NVRTC" if ok else "wrong result")


def check_cpu_kernels() -> Check:
    """The CPU kernel backend: finds g++ or clang++, then builds and runs a
    kernel through it (cached on disk after the first time, so every render's doctor is
    quick). Without a compiler kernels run on the NumPy reference, on one core."""
    import numpy as np

    from phasor.render.cpu import backend as cpu

    c = cpu.find_compiler()
    if c is None:
        other = cpu.unsupported_compilers()
        note = f" ({', '.join(other)} found, not supported)" if other else ""
        return Check("cpu_kernels", None,
                     f"no g++ or clang++{note}: kernels run on NumPy, on one core")  # fmt: skip
    host = cpu.default_cpu()
    assert host is not None
    n = 4096
    x = np.arange(n, dtype=np.float32)
    y = np.empty_like(x)
    fn = host.compile(_ADD_KERNEL, "phasor_doctor_axpb")
    host.launch(fn, n, (x, y, np.float32(2.0), np.float32(1.0), np.int32(n)))
    ok = bool(np.array_equal(y, 2.0 * x + 1.0))
    how = "OpenMP" if c.openmp else "std::thread"
    data = {"compiler": c.name, "version": c.version, "path": c.path, "openmp": c.openmp,
            "threads": host.threads, "cores": cpu.cores()}  # fmt: skip
    detail = f"{c.version}; {how}, {host.threads} threads of {cpu.cores()} cores"
    return Check("cpu_kernels", ok, detail if ok else f"{detail}: wrong result", data)


def check_nvidia_smi() -> Check:
    """Fallback GPU identification when CuPy is missing or broken."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return Check("nvidia_smi", None, "nvidia-smi not found")
    out = subprocess.run(
        [exe, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
        creationflags=NO_WINDOW,
    )
    if out.returncode != 0:
        return Check("nvidia_smi", False, out.stderr.strip() or "nvidia-smi failed")
    line = out.stdout.strip().splitlines()[0]
    return Check("nvidia_smi", True, line, {"raw": line})


# ------------------------------------------------------------------ OpenGL

_TRI_VS = """
#version 430
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""
_TRI_FS = """
#version 430
out vec4 color;
void main() { color = vec4(1.0, 0.5, 0.25, 1.0); }
"""
_ID_FS = """
#version 430
layout(location = 0) out uint obj_id;
uniform uint u_id;
void main() { obj_id = u_id; }
"""


def check_opengl() -> Check:

    from phasor.render.glcontext import create_standalone_context

    ctx = create_standalone_context()  # its own context: the doctor must not disturb renderers
    try:
        with ctx:
            return _exercise_opengl(ctx)
    finally:
        ctx.release()


def _exercise_opengl(ctx) -> Check:
    import numpy as np

    info = ctx.info
    data = {
        "renderer": info.get("GL_RENDERER"),
        "vendor": info.get("GL_VENDOR"),
        "version": info.get("GL_VERSION"),
        "version_code": ctx.version_code,
        "software": "llvmpipe" in str(info.get("GL_RENDERER", "")).lower(),
    }
    size = 64
    tri = np.array([-0.8, -0.8, 0.8, -0.8, 0.0, 0.8], dtype="f4")
    vbo = ctx.buffer(tri.tobytes())

    # Color pass analogue: RGBA16F target (contract 4).
    prog = ctx.program(vertex_shader=_TRI_VS, fragment_shader=_TRI_FS)
    vao = ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])
    color = ctx.texture((size, size), 4, dtype="f2")
    fbo = ctx.framebuffer(color_attachments=[color])
    fbo.use()
    fbo.clear(0, 0, 0, 0)
    vao.render()
    px = np.frombuffer(fbo.read(components=4, dtype="f2"), dtype=np.float16).reshape(size, size, 4)
    center_ok = np.allclose(px[size // 2, size // 2], [1.0, 0.5, 0.25, 1.0], atol=1e-2)
    corner_ok = np.allclose(px[0, 0], 0)

    # ID pass analogue: R32UI target + depth, read back and counted.
    id_prog = ctx.program(vertex_shader=_TRI_VS, fragment_shader=_ID_FS)
    id_prog["u_id"].value = 7
    id_vao = ctx.vertex_array(id_prog, [(vbo, "2f", "in_pos")])
    id_tex = ctx.texture((size, size), 1, dtype="u4")
    depth = ctx.depth_renderbuffer((size, size))
    id_fbo = ctx.framebuffer(color_attachments=[id_tex], depth_attachment=depth)
    id_fbo.use()
    id_fbo.clear()
    id_vao.render()
    ids = np.frombuffer(id_fbo.read(components=1, dtype="u4"), dtype=np.uint32)
    counts = np.bincount(ids, minlength=8)
    id_ok = counts[7] > 0 and counts[0] > 0 and counts[7] + counts[0] == ids.size

    ok = bool(center_ok and corner_ok and id_ok)
    data["triangle_ok"] = bool(center_ok and corner_ok)
    data["id_pass_ok"] = bool(id_ok)
    detail = f"{data['renderer']} (GL {data['version_code']})"
    if data["software"]:
        detail += " [software renderer]"
    if not ok:
        detail += f"; triangle_ok={data['triangle_ok']} id_pass_ok={data['id_pass_ok']}"
    return Check("opengl", ok, detail, data)


# ------------------------------------------------------------------ encoders


def _pyav_encode(codec: str, frames: int = 8) -> None:
    import av
    import numpy as np

    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as container:
        stream = container.add_stream(codec, rate=30)
        stream.width, stream.height, stream.pix_fmt = 256, 144, "yuv420p"
        for i in range(frames):
            img = np.full((144, 256, 3), (i * 30) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if buf.tell() == 0:
        raise RuntimeError("encoder produced no bytes")


def check_pyav() -> Check:
    try:
        import av
    except ImportError:
        return Check("pyav", None, "PyAV not installed (pip install phasor[encode])")
    working: list[str] = []
    errors: dict[str, str] = {}
    for codec in PYAV_ENCODERS:
        try:
            _pyav_encode(codec)
            working.append(codec)
        except Exception as e:
            errors[codec] = f"{type(e).__name__}: {e}"[:200]
    return Check(
        "pyav",
        bool(working),
        f"PyAV {av.__version__}: working " + (", ".join(working) or "none"),
        {"version": av.__version__, "working": working, "errors": errors},
    )


def check_ffmpeg() -> Check:
    exe = shutil.which("ffmpeg")
    if not exe:
        return Check("ffmpeg", None, "no system ffmpeg on PATH")
    ver = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, text=True,
                         creationflags=NO_WINDOW)  # fmt: skip
    m = re.search(r"ffmpeg version (\S+)", ver.stdout)
    working: list[str] = []
    for codec in FFMPEG_ENCODERS:
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
               "color=c=black:s=256x144:d=0.3", "-c:v", codec, "-f", "null", "-"]  # fmt: skip
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                               creationflags=NO_WINDOW)  # fmt: skip
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0:
            working.append(codec)
    return Check(
        "ffmpeg",
        bool(working),
        f"ffmpeg {m.group(1) if m else '?'} at {exe}: working " + (", ".join(working) or "none"),
        {"path": exe, "working": working},
    )


def pick_encoder(pyav: Check | None, ffmpeg: Check | None) -> str | None:
    """Probe order: PyAV NVENC, then system ffmpeg (NVENC first), then PyAV libx264."""
    pyav_ok = pyav.data.get("working", []) if pyav and pyav.ok else []
    ff_ok = ffmpeg.data.get("working", []) if ffmpeg and ffmpeg.ok else []
    for codec in ("h264_nvenc", "hevc_nvenc"):
        if codec in pyav_ok:
            return f"pyav:{codec}"
    for codec in FFMPEG_ENCODERS:
        if codec in ff_ok:
            return f"ffmpeg:{codec}"
    if "libx264" in pyav_ok:
        return "pyav:libx264"
    return None


# ------------------------------------------------------------------ tier


def compute_tier(checks: list[Check]) -> str:
    """A: CUDA kernels run and GL 4.3 works. B: GL 4.3 only. C: neither usable."""
    by = {c.name: c for c in checks}
    gl = by.get("opengl")
    k = by.get("rawkernel")
    if not (gl and gl.ok):
        return "C"
    if k and k.ok:
        return "A"
    return "B"


_REPORT: DoctorReport | None = None


def cached_report() -> DoctorReport:
    """The doctor's report, probed once per process. A render needs only the tier and the
    encoder, and the machine does not change under a running program; probing again cost
    every export (and every desktop-app render) the CUDA, OpenGL and ffmpeg checks.
    ``phasor doctor`` and the app's Doctor still probe fresh."""
    global _REPORT
    if _REPORT is None:
        _REPORT = run_doctor()
    return _REPORT


def run_doctor() -> DoctorReport:
    checks: list[Check] = [_guard("python", check_python)]
    cuda = _guard("cuda", check_cuda)
    checks.append(cuda)
    if cuda.ok:
        checks.append(_guard("nvrtc", check_nvrtc))
        checks.append(_guard("rawkernel", check_rawkernel))
    else:
        checks.append(Check("nvrtc", None, "skipped: no CUDA"))
        checks.append(Check("rawkernel", None, "skipped: no CUDA"))
        checks.append(_guard("nvidia_smi", check_nvidia_smi))
        checks.append(_guard("cpu_kernels", check_cpu_kernels))
    checks.append(_guard("opengl", check_opengl))
    pyav = _guard("pyav", check_pyav)
    ffmpeg = _guard("ffmpeg", check_ffmpeg)
    checks += [pyav, ffmpeg]
    return DoctorReport(checks, compute_tier(checks), pick_encoder(pyav, ffmpeg))


TIER_TEXT = {
    "A": "everything (CUDA kernels + OpenGL)",
    "B": "OpenGL only: kernels fall back to NumPy, final renders are slow",
    "C": "no rendering: inspector (analytic), lint and timeline only",
}
TIER_B_CPU = "OpenGL, no CUDA: kernels run on the CPU ({threads} threads, {compiler})"


def tier_text(report: DoctorReport) -> str:
    """What the tier means here. Tier B says whether its kernels run on the CPU backend
    (a C++ compiler was found and works) or on the NumPy reference."""
    cpu = report.get("cpu_kernels")
    if report.tier == "B" and cpu is not None and cpu.ok:
        return TIER_B_CPU.format(threads=cpu.data["threads"], compiler=cpu.data["compiler"])
    return TIER_TEXT[report.tier]


def format_report(report: DoctorReport) -> str:
    mark = {True: "ok  ", False: "FAIL", None: "skip"}
    lines = ["phasor doctor", ""]
    for c in report.checks:
        lines.append(f"  [{mark[c.ok]}] {c.name:<11} {c.detail}")
    lines += [
        "",
        f"  encoder: {report.encoder or 'none found'}",
        f"  tier:    {report.tier} - {tier_text(report)}",
    ]
    return "\n".join(lines)
