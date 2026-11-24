from __future__ import annotations

import json

import pytest

from phasor.cli import EXIT_NOT_IMPLEMENTED, PLANNED, main
from phasor.render.doctor import (
    Check,
    DoctorReport,
    check_opengl,
    check_pyav,
    check_rawkernel,
    compute_tier,
    format_report,
    pick_encoder,
    run_doctor,
)

# ---------------------------------------------------------------- pure logic


def _checks(gl: bool | None, kernel: bool | None) -> list[Check]:
    return [Check("opengl", gl, ""), Check("rawkernel", kernel, "")]


@pytest.mark.parametrize(
    ("gl", "kernel", "tier"),
    [
        (True, True, "A"),
        (True, None, "B"),
        (True, False, "B"),
        (False, True, "C"),  # CUDA without GL cannot render anything
        (None, None, "C"),
        (False, False, "C"),
    ],
)
def test_tier(gl, kernel, tier):
    assert compute_tier(_checks(gl, kernel)) == tier


def _enc(name: str, working: list[str]) -> Check:
    return Check(name, bool(working), "", {"working": working})


@pytest.mark.parametrize(
    ("pyav", "ffmpeg", "want"),
    [
        (["hevc_nvenc", "h264_nvenc", "libx264"], ["libx264"], "pyav:h264_nvenc"),
        (["hevc_nvenc", "libx264"], [], "pyav:hevc_nvenc"),
        (["libx264"], ["hevc_nvenc", "h264_nvenc", "libx264"], "ffmpeg:h264_nvenc"),
        (["h264_nvenc", "libx264"], [], "pyav:h264_nvenc"),
        (["libx264"], ["h264_nvenc", "libx264"], "ffmpeg:h264_nvenc"),
        (["libx264"], ["libx264"], "ffmpeg:libx264"),
        (["libx264"], [], "pyav:libx264"),
        ([], [], None),
    ],
)
def test_encoder_probe_order(pyav, ffmpeg, want):
    assert pick_encoder(_enc("pyav", pyav), _enc("ffmpeg", ffmpeg)) == want


def test_report_serializes_and_formats():
    r = DoctorReport([Check("python", True, "3.12")], "C", None)
    assert json.loads(json.dumps(r.to_dict()))["tier"] == "C"
    assert "tier:    C" in format_report(r)


# ---------------------------------------------------------------- real probes


def test_run_doctor_never_raises():
    report = run_doctor()
    assert report.tier in {"A", "B", "C"}
    names = [c.name for c in report.checks]
    for required in ["python", "cuda", "nvrtc", "rawkernel", "opengl", "pyav", "ffmpeg"]:
        assert required in names


@pytest.mark.gl
def test_opengl_triangle_and_id_pass():
    c = check_opengl()
    assert c.ok, c.detail
    assert c.data["version_code"] >= 430
    assert c.data["triangle_ok"] and c.data["id_pass_ok"]


@pytest.mark.gpu
def test_rawkernel_runs():
    c = check_rawkernel()
    assert c.ok, c.detail


def test_pyav_reports_only_encoders_that_really_encode():
    pytest.importorskip("av")
    c = check_pyav()
    # libx264 ships in the PyAV wheels, so it must work everywhere.
    assert "libx264" in c.data["working"], c.data
    # NVENC entries must have been proven by encoding, not by codec listing.
    for codec in c.data["working"]:
        assert codec not in c.data["errors"]


# ---------------------------------------------------------------- CLI


def test_cli_doctor_json(capsys):
    assert main(["doctor", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["tier"] in {"A", "B", "C"}


def test_cli_doctor_require_tier(capsys):
    tier = run_doctor().tier
    assert main(["doctor", "--require", "C"]) == 0
    rc = main(["doctor", "--require", "A"])
    assert rc == (0 if tier == "A" else 1)


@pytest.mark.parametrize("command", sorted(PLANNED))
def test_planned_commands_say_which_milestone(command, capsys):
    assert main([command]) == EXIT_NOT_IMPLEMENTED
    assert f"Milestone {PLANNED[command][0]}" in capsys.readouterr().err
