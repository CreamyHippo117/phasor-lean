"""Your taste as data.

Every time a person (or the critic) picks one of several variants, the choice is appended to
``style/choices.jsonl``: one JSON object per line, the options offered and the winner. Nothing
here changes a theme by itself; ``tune`` reads the log and says what it shows, so defaults can
be moved toward your eye on purpose.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("style") / "choices.jsonl"


def record(
    path: str | Path,
    *,
    scene: str,
    beat: str,
    options: list[str],
    winner: str,
    by: str = "person",
    note: str = "",
) -> dict[str, Any]:
    """Append one choice. ``winner`` must be one of ``options``."""
    if winner not in options:
        raise ValueError(f"winner {winner!r} is not one of the options {options}")
    entry = {"scene": scene, "beat": beat, "options": list(options), "winner": winner,
             "by": by, "note": note}  # fmt: skip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def read(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: not valid JSON ({e.msg})") from None
    return out


def tune(path: str | Path, min_choices: int = 3) -> dict[str, Any]:
    """What the log says: for each option name, how often it was offered and won, and which
    one you favor (a win rate over at least ``min_choices`` appearances, and more than a fair
    share). ``favored`` is None until there is enough to say."""
    log = read(path)
    offered: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    for c in log:
        offered.update(c["options"])
        wins[c["winner"]] += 1
    rates = {
        name: {
            "offered": offered[name],
            "won": wins[name],
            "rate": round(wins[name] / offered[name], 3),
        }
        for name in sorted(offered)
    }
    fair = {name: 1.0 / max(1, max(len(c["options"]) for c in log if name in c["options"]))
            for name in offered}  # fmt: skip
    ranked = sorted(
        (n for n, r in rates.items() if r["offered"] >= min_choices and r["rate"] > fair[n] + 1e-9),
        key=lambda n: (-rates[n]["rate"], n),
    )
    return {"choices": len(log), "options": rates, "favored": ranked[0] if ranked else None}
