"""Issues: the one shape for validation errors, lint findings and patch failures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    severity: str = "error"  # "error" | "warning"
    ids: tuple[str, ...] = ()
    path: str | None = None
    fix_hint: str | None = None
    times: tuple[float, ...] = ()  # sample times (seconds) at which a lint rule fired
    # Patch ops that would fix it, for ``Engine.autofix``. Not part of the issue's identity.
    fix: tuple[dict[str, Any], ...] = field(default=(), compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    def __str__(self) -> str:
        where = f" [{self.path or ', '.join(self.ids)}]" if (self.path or self.ids) else ""
        return f"{self.code}{where}: {self.message}"


class DocumentError(ValueError):
    """The document (or a patch to it) is invalid. Carries every issue found, not just the first."""

    def __init__(self, issues: list[Issue] | Issue):
        self.issues: list[Issue] = [issues] if isinstance(issues, Issue) else list(issues)
        super().__init__("; ".join(str(i) for i in self.issues))
