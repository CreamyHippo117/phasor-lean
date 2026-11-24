"""Canonical files: ``name.phasor.json`` (+ optional ``name.overrides.json``).

Sorted keys, fixed indentation and a trailing newline keep git diffs clean (contract 2).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from phasor.core.issues import DocumentError, Issue
from phasor.core.migrate import migrate
from phasor.core.schema import Document, Overrides

DOC_SUFFIX = ".phasor.json"
OVERRIDES_SUFFIX = ".overrides.json"


def schema_issues(e: ValidationError) -> list[Issue]:
    return [
        Issue("schema", err["msg"], path=".".join(str(p) for p in err["loc"]) or None)
        for err in e.errors()
    ]


def dumps(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def parse_document(raw: dict) -> Document:
    try:
        return Document.model_validate(migrate(raw))
    except ValidationError as e:
        raise DocumentError(schema_issues(e)) from e


def overrides_path(doc_path: str | Path) -> Path:
    p = Path(doc_path)
    stem = p.name[: -len(DOC_SUFFIX)] if p.name.endswith(DOC_SUFFIX) else p.stem
    return p.with_name(stem + OVERRIDES_SUFFIX)


def load_document(path: str | Path) -> Document:
    return parse_document(json.loads(Path(path).read_text(encoding="utf-8")))


def load_overrides(path: str | Path) -> Overrides:
    p = Path(path)
    if not p.exists():
        return Overrides()
    try:
        return Overrides.model_validate(json.loads(p.read_text(encoding="utf-8")))
    except ValidationError as e:
        raise DocumentError(schema_issues(e)) from e


def save(path: str | Path, model: BaseModel) -> None:
    Path(path).write_text(dumps(model), encoding="utf-8", newline="\n")
