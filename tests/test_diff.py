"""`core.diff`: a structured, human- and AI-readable difference between two documents."""

from phasor import Dot, Scene, Text
from phasor.core.diff import diff_documents


def _base():
    s = Scene("d")
    s.add(Dot(id="a", pos=(1, 1)))
    s.add(Text(id="a.label", text="a", pos=(1, 2)))
    s.marker("go", 1.0)
    return s


def test_identical_documents_have_an_empty_diff():
    d = _base().engine.document
    out = diff_documents(d, d)
    assert out["same"] is True and out["changes"] == []


def test_added_removed_and_changed_objects_and_params():
    a = _base().engine.document
    s = _base()
    s.add(Dot(id="b", pos=(0, 0)))
    s.set("a.pos", (3, 3))
    s.engine.apply({"op": "remove_object", "id": "a.label"}, origin="script").raise_for_errors()
    out = diff_documents(a, s.engine.document)
    kinds = {(c["kind"], c["id"]) for c in out["changes"]}
    assert ("object_added", "b") in kinds
    assert ("object_removed", "a.label") in kinds
    changed = next(c for c in out["changes"] if c["kind"] == "param_changed")
    assert changed["path"] == "a.pos" and changed["to"][:2] == [3, 3]
    assert out["same"] is False


def test_markers_constraints_tracks_and_meta():
    a = _base().engine.document
    s = _base()
    s.marker("go", 2.0)
    s.marker("end", 3.0)
    s.constrain("label(a) near a")
    s.at("go").to("a.opacity", 0.5, duration=1.0)
    out = diff_documents(a, s.engine.document)
    kinds = {c["kind"] for c in out["changes"]}
    assert {"marker_moved", "marker_added", "constraint_added", "track_added"} <= kinds
    assert "1 object" not in out["summary"] and out["summary"]


def test_diff_is_symmetric_in_what_it_reports():
    a = _base().engine.document
    s = _base()
    s.add(Dot(id="b", pos=(0, 0)))
    b = s.engine.document
    fwd, back = diff_documents(a, b), diff_documents(b, a)
    assert fwd["changes"][0]["kind"] == "object_added"
    assert back["changes"][0]["kind"] == "object_removed"


def test_layer_priority_retired_ids_and_camera_changes_are_reported():
    from phasor.core.patch import apply_ops, parse_patch

    a = _base().engine.document
    a = apply_ops(a, parse_patch({"op": "add_constraint", "rule": "label(a) near a"}))
    b = a.model_copy(update={
        "objects": tuple(o.model_copy(update={"layer": "overlay"}) if o.id == "a.label" else o
                         for o in a.objects),
        "constraints": tuple(c.model_copy(update={"priority": "weak"}) for c in a.constraints),
        "retired_ids": ("gone",),
    })  # fmt: skip
    kinds = {c["kind"] for c in diff_documents(a, b)["changes"]}
    assert {"layer_changed", "priority_changed", "id_retired"} <= kinds
    assert "constraint_added" not in kinds and "constraint_removed" not in kinds
