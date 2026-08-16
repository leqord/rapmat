"""Paint an RGB framebuffer into a urwid canvas.
"""

import time
from typing import Callable, Optional

import numpy as np
import urwid

GLYPH = "▀"
_GLYPH_BYTES = GLYPH.encode("utf-8")

_BYTES_PER_CELL = len(_GLYPH_BYTES)

_MAX_ATTR_CACHE = 16384

MIN_COLS = 20
MIN_ROWS = 6

FrameSource = Callable[[int, int], Optional[np.ndarray]]


def glyph_is_renderable() -> bool:
    try:
        encoded, _cs = urwid.util.apply_target_encoding(GLYPH)
    except Exception:
        return False
    return encoded == _GLYPH_BYTES


class AttrSpecCache:

    def __init__(self, colors: int = 2 ** 24, maxsize: int = _MAX_ATTR_CACHE) -> None:
        self._colors = colors
        self._maxsize = maxsize
        self._cache: dict[int, urwid.AttrSpec] = {}

    @property
    def colors(self) -> int:
        return self._colors

    def set_colors(self, colors: int) -> None:
        if colors != self._colors:
            self._colors = colors
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, key: int) -> urwid.AttrSpec:
        spec = self._cache.get(key)
        if spec is None:
            if len(self._cache) >= self._maxsize:
                self._cache.clear()
            spec = urwid.AttrSpec(
                "#%06x" % (key >> 24), "#%06x" % (key & 0xFFFFFF), self._colors
            )
            self._cache[key] = spec
        return spec


def frame_to_canvas(fb: np.ndarray, cache: AttrSpecCache) -> urwid.TextCanvas:
    height, width, _ = fb.shape
    rows = height // 2
    if rows == 0 or width == 0:
        return urwid.TextCanvas([b""] * rows, maxcol=width, check_width=False)

    top = fb[0:2 * rows:2].astype(np.int64)
    bot = fb[1:2 * rows:2].astype(np.int64)
    packed_top = (top[..., 0] << 16) | (top[..., 1] << 8) | top[..., 2]
    packed_bot = (bot[..., 0] << 16) | (bot[..., 1] << 8) | bot[..., 2]
    keys = (packed_top << 24) | packed_bot

    line = _GLYPH_BYTES * width
    text: list[bytes] = []
    attr: list[list[tuple[urwid.AttrSpec, int]]] = []

    change = np.empty(width, dtype=bool)
    for r in range(rows):
        row = keys[r]
        change[0] = True
        np.not_equal(row[1:], row[:-1], out=change[1:])
        starts = np.flatnonzero(change)
        lengths = np.diff(np.append(starts, width))
        flat = row.tolist()
        attr.append([
            (cache[flat[s]], int(n) * _BYTES_PER_CELL)
            for s, n in zip(starts.tolist(), lengths.tolist())
        ])
        text.append(line)

    return urwid.TextCanvas(text=text, attr=attr, maxcol=width, check_width=False)


class HalfBlockCanvas(urwid.Widget):

    _sizing = frozenset([urwid.BOX])
    _selectable = False
    ignore_focus = True

    def __init__(self, frame_source: FrameSource, *, colors: int = 2 ** 24) -> None:
        super().__init__()
        self._frame_source = frame_source
        self._cache = AttrSpecCache(colors)
        self._message = ""
        self._text = urwid.Text("", align=urwid.CENTER)
        self._filler = urwid.Filler(self._text, valign=urwid.MIDDLE)
        self.last_render_seconds = 0.0

    def mark_dirty(self) -> None:
        self._invalidate()

    def set_message(self, text: str) -> None:
        if text != self._message:
            self._message = text
            self._invalidate()

    def set_colors(self, colors: int) -> None:
        self._cache.set_colors(colors)
        self._invalidate()

    def render(self, size, focus: bool = False):
        maxcol, maxrow = size
        if maxcol <= 0 or maxrow <= 0:
            return urwid.SolidCanvas(" ", max(maxcol, 0), max(maxrow, 0))
        if self._message:
            return self._fallback(size, self._message)
        if maxcol < MIN_COLS or maxrow < MIN_ROWS:
            return self._fallback(size, "Terminal too small for the 3D view")

        started = time.perf_counter()
        try:
            fb = self._frame_source(maxcol, maxrow * 2)
        except Exception as exc:
            return self._fallback(size, f"Render failed: {exc}")
        if fb is None or fb.shape[0] < 2 * maxrow or fb.shape[1] != maxcol:
            return self._fallback(size, "Nothing to display")

        canvas = frame_to_canvas(fb[:2 * maxrow], self._cache)
        self.last_render_seconds = time.perf_counter() - started
        return canvas

    def _fallback(self, size, text: str):
        self._text.set_text(("details", text))
        return self._filler.render(size, focus=False)
