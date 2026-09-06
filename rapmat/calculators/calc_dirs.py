import re
from pathlib import Path, PurePosixPath

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_PART = 120

PRUNE_AFTER_CALC = (
    "WAVECAR",
    "WAVECAR.h5",
    "TMPCAR",
    "CHGCAR",
    "CHG",
    "PROCAR",
    "DOSCAR",
)


def sanitize_label(label) -> PurePosixPath:
    parts = []
    for raw in re.split(r"[/\\]", str(label)):
        part = _UNSAFE.sub("_", raw.strip())[:_MAX_PART]
        if part and part.strip("."):
            parts.append(part)
    return PurePosixPath(*(parts or ["_"]))


def unlink_quietly(path: Path) -> int:
    try:
        path.unlink()
        return 1
    except OSError:
        return 0


def prune_calculation_dir(directory) -> int:
    if directory is None:
        return 0
    dirpath = Path(directory)
    if not dirpath.is_dir():
        return 0
    return sum(unlink_quietly(dirpath / name) for name in PRUNE_AFTER_CALC)


class CalcDirAllocator:

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root is not None else None
        self._label: str | None = None
        self._seq = 0
        self._previous: Path | None = None

    @property
    def active(self) -> bool:
        return self._root is not None

    @property
    def scope_path(self) -> Path | None:
        if self._root is None:
            return None
        if self._label is None:
            return self._root
        return self._root.joinpath(*sanitize_label(self._label).parts)

    def set_label(self, label) -> None:
        label = None if label is None else str(label)
        # NOTE: idempotent on purpose
        if label == self._label:
            return
        self._label = label
        self._seq = 0

    def next(self) -> Path | None:
        if self._root is None:
            return None

        self._seq += 1
        allocated = self.scope_path / f"{self._seq:04d}"

        prune_calculation_dir(self._previous)
        self._previous = allocated

        return allocated

    def finalize(self) -> None:
        prune_calculation_dir(self._previous)
        self._previous = None
