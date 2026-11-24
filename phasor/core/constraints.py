"""The constraint language: one rule string -> clauses.

    label(pole) near pole
    title above plot gap 0.3
    a align_x b
    on_screen(title, 0.2) and no_overlap(a, b, c)

This module only parses. ``phasor.layout`` gives the clauses meaning and solves them.
Numbers are in the units of the object's layer (world units, or frame heights on the overlay).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from phasor.core.ids import ID_RE

# relation -> takes an optional "gap <number>"
RELATIONS = {
    "above": True,
    "below": True,
    "left_of": True,
    "right_of": True,
    "near": True,
    "align_x": False,
    "align_y": False,
    "inside": False,
    "outside": True,  # not overlapping: on whichever side needs the least movement
    "tangent_to": False,  # touching the outside edge of the target, on the nearest side
    "attach_to": True,  # a named spot on the target: ``a attach_to b at top``
}
# Spots every object has (on its screen box); objects may add their own feature names
# (an arrow's ``tip``, a line's ``end``).
BOX_ANCHORS = {
    "center": (0, 0), "top": (0, 1), "bottom": (0, -1), "left": (-1, 0), "right": (1, 0),
    "top_left": (-1, 1), "top_right": (1, 1), "bottom_left": (-1, -1), "bottom_right": (1, -1),
}  # fmt: skip
# guard -> (min ids, max ids or None for any number, takes an optional trailing number)
GUARDS = {
    "visible_in_3d": (1, 1, False),  # not hidden by what is in front of it (checked by lint)
    "no_overlap": (2, None, False),
    "on_screen": (1, 1, True),
    "min_size_px": (1, 1, True),
    "readable": (1, 1, False),
}
# camera goal -> (min ids, max ids or None, takes a trailing number). Written
# ``camera keep_visible(a, b)``: the camera solver moves the camera to satisfy them.
CAMERA = "camera"
CAMERA_GOALS = {
    "keep_visible": (1, None, False),
    "center": (1, 1, False),
    "thirds": (1, 1, False),
    "avoid_occlusion": (1, None, False),
    "min_size": (1, 1, True),  # share of the frame height its bounding sphere should span
}
PRIORITIES = ("required", "strong", "medium", "weak")
LABEL_SUFFIX = "label"

_TOKEN = re.compile(rf"\s*(?:(?P<num>-?\d+(?:\.\d+)?)|(?P<id>{ID_RE.pattern})|(?P<sym>[(),]))")


class RuleError(ValueError):
    pass


@dataclass(frozen=True)
class Relation:
    kind: str
    subject: str  # the object that moves
    target: str  # the object it is placed against
    gap: float | None = None
    anchor: str | None = None  # attach_to: the spot on the target (``at top``)

    @property
    def ids(self) -> tuple[str, ...]:
        return (self.subject, self.target)


@dataclass(frozen=True)
class Guard:
    kind: str
    ids: tuple[str, ...]
    value: float | None = None


@dataclass(frozen=True)
class Goal:
    """A camera goal: what the camera solver should achieve, not an angle."""

    kind: str
    ids: tuple[str, ...]
    value: float | None = None


Clause = Relation | Guard | Goal


def label_id(owner: str) -> str:
    """The label of ``pole`` is the child object ``pole.label``; the id says who it labels."""
    return f"{owner}.{LABEL_SUFFIX}"


def normalize_rule(rule: str) -> str:
    return " ".join(rule.split())


def _tokens(rule: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    while pos < len(rule):
        if rule[pos:].strip() == "":
            break
        m = _TOKEN.match(rule, pos)
        if not m:
            raise RuleError(f"cannot read {rule[pos:].strip()[:12]!r} in constraint {rule!r}")
        out.append((m.lastgroup, m.group(m.lastgroup)))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, rule: str) -> None:
        self.rule, self.toks, self.i = rule, _tokens(rule), 0

    def fail(self, msg: str) -> RuleError:
        return RuleError(f"{msg} in constraint {self.rule!r}")

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, kind: str | None = None, value: str | None = None) -> str:
        tok = self.peek()
        if tok is None or (kind and tok[0] != kind) or (value and tok[1] != value):
            want = value or {"num": "a number", "id": "an object id", "sym": "a symbol"}[kind]
            got = "the end" if tok is None else repr(tok[1])
            raise self.fail(f"expected {want}, got {got}")
        self.i += 1
        return tok[1]

    def at(self, value: str) -> bool:
        return self.peek() is not None and self.peek()[1] == value

    def ref(self) -> str:
        name = self.take("id")
        if name == LABEL_SUFFIX and self.at("("):
            self.take("sym", "(")
            owner = self.take("id")
            self.take("sym", ")")
            return label_id(owner)
        return name

    def clause(self) -> Clause:
        tok = self.peek()
        if tok == ("id", CAMERA) and self.i + 1 < len(self.toks):
            nxt = self.toks[self.i + 1][1]
            call = self.i + 2 < len(self.toks) and self.toks[self.i + 2][1] == "("
            if nxt in CAMERA_GOALS or (call and nxt not in GUARDS):
                self.take("id", CAMERA)
                g = self.guard(CAMERA_GOALS)
                return Goal(g.kind, g.ids, g.value)
            if nxt in GUARDS or nxt in RELATIONS:
                raise self.fail(f"the camera takes goals ({', '.join(sorted(CAMERA_GOALS))}), "
                                f"not {nxt!r}")  # fmt: skip
        if tok and tok[0] == "id" and tok[1] in GUARDS and self._call_follows():
            return self.guard()
        subject = self.ref()
        kind = self.take("id")
        if kind not in RELATIONS:
            raise self.fail(f"unknown relation {kind!r} (use one of {sorted(RELATIONS)})")
        target = self.ref()
        anchor = None
        if kind == "attach_to":
            if not self.at("at"):
                raise self.fail("attach_to needs a spot: 'a attach_to b at top'")
            self.take("id", "at")
            anchor = self.take("id")
        gap = None
        if self.at("gap"):
            if not RELATIONS[kind]:
                raise self.fail(f"{kind!r} takes no gap")
            self.take("id", "gap")
            gap = float(self.take("num"))
            if gap < 0:
                raise self.fail("gap must not be negative")
        if subject == target:
            raise self.fail(f"{subject!r} cannot be constrained against itself")
        return Relation(kind, subject, target, gap, anchor)

    def _call_follows(self) -> bool:
        return self.i + 1 < len(self.toks) and self.toks[self.i + 1][1] == "("

    def guard(self, table: dict = GUARDS) -> Guard:
        kind = self.take("id")
        if kind not in table:
            raise self.fail(f"unknown camera goal {kind!r} (use one of {sorted(table)})")
        lo, hi, numbered = table[kind]
        self.take("sym", "(")
        ids: list[str] = []
        value = None
        while True:
            tok = self.peek()
            if tok and tok[0] == "num":
                if not numbered:
                    raise self.fail(f"{kind} takes no number")
                value = float(self.take("num"))
                self.take("sym", ")")
                break
            ids.append(self.ref())
            if self.at(")"):
                self.take("sym", ")")
                break
            self.take("sym", ",")
        if len(ids) < lo or (hi is not None and len(ids) > hi) or len(set(ids)) != len(ids):
            n = f"{lo}" if hi == lo else f"at least {lo}"
            raise self.fail(f"{kind} needs {n} different object ids")
        if kind == "min_size_px" and value is None:
            raise self.fail("min_size_px needs a size, e.g. min_size_px(dot, 12)")
        if kind == "min_size" and (value is None or not 0 < value <= 1):
            raise self.fail("min_size needs a share of the frame height, e.g. min_size(a, 0.3)")
        if value is not None and value < 0:
            raise self.fail("the number must not be negative")
        return Guard(kind, tuple(ids), value)


def parse_rule(rule: str) -> tuple[Clause, ...]:
    """Parse one rule into its clauses. Raises RuleError with the reason."""
    p = _Parser(rule)
    if not p.toks:
        raise RuleError("empty constraint")
    clauses = [p.clause()]
    while p.peek() is not None:
        p.take("id", "and")
        clauses.append(p.clause())
    return tuple(clauses)


def rule_ids(rule: str) -> tuple[str, ...]:
    """Every object id a rule mentions, in order of first appearance."""
    seen: dict[str, None] = {}
    for c in parse_rule(rule):
        seen.update(dict.fromkeys(c.ids))
    return tuple(seen)
