"""The editor's code view: a document written back as a Python builder script that, run,
builds the same scene."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from phasor import Scene
from phasor.core.diff import diff_documents
from phasor.core.pycode import document_to_python

ROOT = Path(__file__).parents[1]


def _time(t):
    if not hasattr(t, "offset"):
        return round(t, 9)
    return t.model_copy(update={"offset": round(t.offset, 9)})  # a marker and an offset


def plain(doc):
    """The document without what a generated script cannot say: the verb that made a segment
    (a raw ``to`` has none) and float noise in times (t1 is recomputed from a duration)."""
    tracks = {
        p: tuple(
            seg.model_copy(update={"verb": None, "t0": _time(seg.t0), "t1": _time(seg.t1)})
            for seg in segs
        )
        for p, segs in doc.timeline.tracks.items()
    }
    return doc.model_copy(update={"timeline": doc.timeline.model_copy(update={"tracks": tracks})})


def rebuild(doc, tmp_path):
    path = tmp_path / "again.py"
    path.write_text(document_to_python(doc), encoding="utf-8")
    return runpy.run_path(str(path), run_name="x")["s"].engine.document


@pytest.mark.parametrize(
    "example", ["julia_explorer", "basics_3d", "heat_dihedral", "prism_primes"]
)
def test_running_the_generated_script_rebuilds_the_scene(example, tmp_path):
    original = runpy.run_path(str(ROOT / "examples" / f"{example}.py"), run_name="x")[
        "s"
    ].engine.document
    again = rebuild(original, tmp_path)
    d = diff_documents(plain(original), plain(again))
    assert d["same"], d["changes"][:3]


def test_the_code_reads_like_a_script_and_covers_every_kind_of_thing(tmp_path):
    s = Scene("tiny", life="off")
    from phasor import Dot, Line, expr

    s.add(
        Dot(
            id="a", pos=(1, 2), r=expr("0.1 + 0.05 * t"), style={"color": "highlight"}, tags=["key"]
        )
    )
    s.add(Line(id="l", points=[(0, 0, 0), (1, 1, 0)]))
    s.marker("go", 1.5)
    s.at("go", 0.25).to("a.pos", (3, 0, 0), duration=2.0, ease="linear")
    s.constrain("on_screen(a)", priority="medium")
    code = document_to_python(s.engine.document)
    assert code.startswith("from phasor import ") and "s = Scene('tiny'" in code
    assert (
        "Dot(id='a'" in code and "expr('0.1 + 0.05 * t')" in code and "s.marker('go', 1.5)" in code
    )
    assert "s.at('go', 0.25).to('a.pos'" in code and "s.constrain('on_screen(a)'" in code
    assert diff_documents(plain(s.engine.document), plain(rebuild(s.engine.document, tmp_path)))[
        "same"
    ]
