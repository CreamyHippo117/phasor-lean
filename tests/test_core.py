"""core: ids, values, schema, migration, canonical files, object contract, camera math."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from phasor.core import io
from phasor.core.ids import is_valid_id, parent_of, split_path
from phasor.core.issues import DocumentError
from phasor.core.objects import (
    REQUIRED_MEMBERS,
    ObjectBase,
    ParamsBase,
    RegistrationError,
    get_type,
    register,
)
from phasor.core.rng import rng
from phasor.core.schema import SCHEMA_VERSION, Document, ExprSpec, Meta, ObjectSpec, json_schema
from phasor.core.spatial import OrthoCameraParams, ortho_camera, overlay_camera, region_name
from phasor.core.values import lerp, normalize, same_shape


@pytest.mark.parametrize("good", ["a", "pole_i", "res.pole_i", "a1.b2.c3"])
def test_valid_ids(good):
    assert is_valid_id(good)


@pytest.mark.parametrize("bad", ["", "A", "1a", "a-b", "a..b", ".a", "a.", "a b", "a.1"])
def test_invalid_ids(bad):
    assert not is_valid_id(bad)


def test_paths_and_parents():
    assert split_path("res.pole_i.pos") == ("res.pole_i", "pos")
    assert parent_of("res.pole_i") == "res"
    assert parent_of("res") is None
    with pytest.raises(ValueError):
        split_path("nopath")


def test_normalize_values():
    assert normalize([1, 2.5, [3]]) == (1.0, 2.5, (3.0,))
    assert normalize(True) is True
    assert normalize(np.array([1, 2])) == (1.0, 2.0)
    for bad in [float("nan"), float("inf"), None, {"a": 1}, object()]:
        with pytest.raises(ValueError):
            normalize(bad)


def test_lerp_and_shape():
    assert lerp(0.0, 10.0, 0.25) == 2.5
    assert lerp((0.0, 0.0), (2.0, 4.0), 0.5) == (1.0, 2.0)
    assert lerp("a", "b", 0.99) == "a" and lerp("a", "b", 1.0) == "b"
    assert lerp(False, True, 0.5) is False
    assert same_shape((1.0, 2.0), (3.0, 4.0))
    assert not same_shape((1.0, 2.0), (1.0, 2.0, 3.0))
    assert not same_shape(1.0, "x")


def _doc(**kw) -> Document:
    return Document(meta=Meta(title="t"), **kw)


def test_schema_rejects_unknown_keys_and_bad_ids():
    with pytest.raises(ValidationError):
        Document.model_validate({"meta": {"title": "t"}, "surprise": 1})
    with pytest.raises(ValidationError, match="invalid id"):
        _doc(objects=[ObjectSpec(id="Bad", type="dot")])
    with pytest.raises(ValidationError, match="duplicate id"):
        _doc(objects=[ObjectSpec(id="a", type="dot"), ObjectSpec(id="a", type="dot")])
    with pytest.raises(ValidationError, match="duplicate id"):
        _doc(objects=[ObjectSpec(id="camera", type="dot")])  # shares the camera namespace
    with pytest.raises(ValidationError, match="parent"):
        _doc(objects=[ObjectSpec(id="res.pole_i", type="dot")])


def test_document_is_frozen():
    doc = _doc()
    with pytest.raises(ValidationError):
        doc.meta.title = "other"
    with pytest.raises(ValidationError):
        doc.objects = ()


def test_params_parse_constants_and_expressions():
    spec = ObjectSpec(id="a", type="dot", params={"pos": [1, 2], "r": {"expr": "2 * b.r"}})
    assert spec.params["pos"] == (1.0, 2.0)
    assert spec.params["r"] == ExprSpec(expr="2 * b.r")
    with pytest.raises(ValidationError):
        ObjectSpec(id="a", type="dot", params={"r": {"not_expr": 1}})


def test_canonical_file_roundtrip_is_byte_stable(tmp_path):
    doc = _doc(
        objects=[ObjectSpec(id="b", type="dot", params={"r": {"expr": "t"}, "pos": (1, 2, 3)})]
    )
    path = tmp_path / "s.phasor.json"
    io.save(path, doc)
    first = path.read_bytes()
    assert first.endswith(b"\n") and b"\r" not in first
    loaded = io.load_document(path)
    assert loaded == doc
    io.save(path, loaded)
    assert path.read_bytes() == first
    keys = list(json.loads(first))
    assert keys == sorted(keys)


def test_migrate_rejects_missing_and_future_versions():
    with pytest.raises(DocumentError, match="schema_version"):
        io.parse_document({"meta": {"title": "t"}})
    with pytest.raises(DocumentError, match=f"reads up to {SCHEMA_VERSION}"):
        io.parse_document({"schema_version": 999, "meta": {"title": "t"}})


def test_json_schema_exports():
    schema = json_schema()
    assert schema["title"] == "Document" and "objects" in schema["properties"]


def test_overrides_path():
    assert io.overrides_path("a/b/scene.phasor.json").name == "scene.overrides.json"


# ---------------------------------------------------------------- object contract


def _complete_type(name: str) -> dict:
    return {
        "type": name,
        "Params": type("Params", (ParamsBase,), {}),
        "world_bounds": lambda self, state: None,
        "features": lambda self, state: {},
        "semantics": lambda self, state: {},
        "draw": lambda self, ctx, state, pass_: None,
    }


@pytest.mark.parametrize("missing", REQUIRED_MEMBERS)
def test_registration_fails_when_a_contract_member_is_missing(missing):
    members = _complete_type(f"incomplete_{missing.lower()}")
    del members[missing]
    with pytest.raises(RegistrationError, match=missing):
        register(type("Incomplete", (ObjectBase,), members))
    assert get_type(members["type"]) is None


def test_registration_rejects_bad_names_params_and_duplicates():
    with pytest.raises(RegistrationError, match="type"):
        register(type("X", (ObjectBase,), {**_complete_type("x"), "type": "Bad Name"}))
    with pytest.raises(RegistrationError, match="ParamsBase"):
        register(type("X", (ObjectBase,), {**_complete_type("x_params"), "Params": dict}))
    import phasor.domains  # noqa: F401

    with pytest.raises(RegistrationError, match="already registered"):
        register(type("Dot2", (ObjectBase,), _complete_type("dot")))


def test_builtin_types_are_registered():
    import phasor.domains  # noqa: F401

    for name in ("dot", "line", "text"):
        assert get_type(name) is not None


# ---------------------------------------------------------------- spatial


def test_ortho_camera_maps_complex_plane_to_screen():
    cam = ortho_camera(OrthoCameraParams(center=(0, 0, 0), height=8), 16 / 9)
    px = cam.to_screen([[0, 0, 0], [0, 4, 0], [64 / 9, 0, 0], [-64 / 9, -4, 0]], 1920, 1080)
    np.testing.assert_allclose(px, [[960, 540], [960, 0], [1920, 540], [0, 1080]], atol=1e-9)
    # Looking down -z: higher z is nearer the camera.
    assert cam.depth([[0, 0, 1]])[0] < cam.depth([[0, 0, 0]])[0]
    assert cam.project([[0, 0, 1]])[0, 2] < cam.project([[0, 0, 0]])[0, 2]


def test_ortho_camera_follows_center():
    cam = ortho_camera(OrthoCameraParams(center=(2, 1, 0), height=4), 2.0)
    np.testing.assert_allclose(cam.to_screen([[2, 1, 0]], 200, 100), [[100, 50]])


def test_overlay_camera_is_screen_space_in_frame_heights():
    cam = overlay_camera(16 / 9)
    px = cam.to_screen([[0, 0, 0], [16 / 9, 1, 0], [8 / 9, 0.5, 0]], 1920, 1080)
    np.testing.assert_allclose(px, [[0, 0], [1920, 1080], [960, 540]], atol=1e-9)


def test_region_names():
    assert region_name(0.1, 0.1) == "upper-left"
    assert region_name(0.5, 0.5) == "center"
    assert region_name(0.9, 0.99) == "lower-right"
    assert region_name(1.4, -0.2) == "upper-right"  # clamped into the frame


def test_rng_is_deterministic_per_object_and_seed():
    assert rng("a", 1).random() == rng("a", 1).random()
    assert rng("a", 1).random() != rng("a", 2).random()
    assert rng("a", 1).random() != rng("b", 1).random()


@pytest.mark.parametrize("px", range(12, 48, 5))
def test_rasterized_text_never_cuts_off_its_last_glyph(px):
    """A Milestone 4 render showed "= pi" as "= p": glyphs drawn at the nearest whole font
    size ran past a bitmap sized from the reference metrics."""
    from phasor.core.text import aspect, rasterize

    text = "integral around the loop = 2 pi i (-i/2) = pi"
    bitmap = rasterize(text, px)
    assert bitmap.shape == (px, math.ceil(aspect(text) * px))
    inked = np.flatnonzero(bitmap.max(axis=0) > 64)
    without_last = rasterize(text[:-1], px)
    assert inked.max() > np.flatnonzero(without_last.max(axis=0) > 64).max() * 0.98
    assert bitmap.shape[1] - inked.max() < max(3, px * 0.4)  # the "i" is there, at the end


def test_v4_documents_migrate_with_life_off():
    doc = io.parse_document({"schema_version": 4, "meta": {"title": "old"}})
    assert doc.schema_version == SCHEMA_VERSION and doc.meta.life == "off" and doc.meta.seed == 0
