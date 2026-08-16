"""Fullscreen 3D structure viewer.
"""

import math
import time
from typing import Callable, Optional

import urwid

from rapmat.core.entities import ResultRow
from rapmat.core.render3d import Scene, render_frame
from rapmat.core.structure_scene import build_scene, next_style, style_label
from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.theme import color_depth_label
from rapmat.tui.widgets.halfblock import HalfBlockCanvas, glyph_is_renderable

DEFAULT_YAW = math.radians(28.0)
DEFAULT_PITCH = math.radians(-18.0)
ROT_STEP = math.radians(6.0)

ZOOM_STEP = 1.12
ZOOM_MIN = 0.15
ZOOM_MAX = 12.0

SPIN_RATE = 0.9
SPIN_INTERVAL = 1.0 / 24.0
REDRAW_INTERVAL = 1.0 / 30.0

MIN_COLOR_DEPTH = 256


def _signed_degrees(angle: float) -> int:
    degrees = round(math.degrees(angle)) % 360
    return degrees - 360 if degrees > 180 else degrees


def _next_delay(interval: float, render_seconds: float) -> float:
    return max(interval, 1.5 * render_seconds)

_ROTATE = {
    "left": (-ROT_STEP, 0.0),
    "h": (-ROT_STEP, 0.0),
    "right": (ROT_STEP, 0.0),
    "l": (ROT_STEP, 0.0),
    "up": (0.0, -ROT_STEP),
    "k": (0.0, -ROT_STEP),
    "down": (0.0, ROT_STEP),
    "j": (0.0, ROT_STEP),
}

SEP = " | "

_LOW_COLOR_NOTICE = (
    "This terminal supports only 16 colours, so the 3D view cannot work.\n"
    "Use a truecolor terminal."
)

_NO_UNICODE_NOTICE = (
    "This terminal's encoding does not support the half-block glyph, so the\n"
    "3D view cannot work. Switch the terminal to UTF-8."
)


class StructureViewScreen(ScreenBase):
    title = "3D View"

    def __init__(
        self,
        state: "AppState",
        router: "ScreenRouter",
        results: list["ResultRow"],
        index: int = 0,
        on_focus: Optional[Callable[["ResultRow"], None]] = None,
    ) -> None:
        super().__init__(state, router)
        self._results = list(results)
        self._index = max(0, min(index, len(self._results) - 1)) if self._results else 0
        self._on_focus = on_focus

        self._yaw = DEFAULT_YAW
        self._pitch = DEFAULT_PITCH
        self._zoom = 1.0
        self._style = "ball"
        self._show_bonds = True
        self._show_cell = True
        self._supercell = 1

        self._spinning = False
        self._spin_handle = None
        self._spin_last = 0.0
        self._redraw_handle = None

        self._scene: Scene | None = None
        self._scene_error = ""
        self._spg: str | None = None
        self._glyph_ok: bool | None = None
        self._last_key = ""

        self._view: HalfBlockCanvas | None = None
        self._identity = urwid.Text("", wrap="clip")
        self._viewstate = urwid.Text("", wrap="clip")

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    @property
    def breadcrumb_title(self) -> str:
        result = self._result
        return f"3D View: {result.short_id}" if result is not None else "3D View"

    @property
    def _result(self) -> "ResultRow | None":
        if not self._results:
            return None
        return self._results[self._index]

    @property
    def _supported(self) -> bool:
        if self._glyph_ok is None:
            self._glyph_ok = glyph_is_renderable()
        return self._state.color_depth >= MIN_COLOR_DEPTH and self._glyph_ok

    @property
    def _renderable(self) -> bool:
        return (
            self._supported
            and self._scene is not None
            and self._scene.n_atoms > 0
        )

    def build(self) -> urwid.Widget:
        self._view = HalfBlockCanvas(
            self._render_frame, colors=self._state.color_depth
        )
        if self._state.color_depth < MIN_COLOR_DEPTH:
            self._view.set_message(_LOW_COLOR_NOTICE)
        elif not self._supported:
            self._view.set_message(_NO_UNICODE_NOTICE)
        else:
            self._rebuild_scene()

        self._refresh_info()
        body = urwid.Pile([
            ("pack", urwid.AttrMap(self._identity, "col_header")),
            ("pack", self._viewstate),
            ("weight", 1, self._view),
        ])
        self.refresh_footer()
        return urwid.Frame(body=body)

    # ------------------------------------------------------------------ #
    #  Scene
    # ------------------------------------------------------------------ #

    def _rebuild_scene(self) -> None:
        result = self._result
        try:
            self._scene = build_scene(
                result.atoms if result is not None else None,
                style=self._style,
                show_bonds=self._show_bonds,
                supercell=self._supercell,
            )
            self._scene_error = ""
        except Exception as exc:
            self._scene = None
            self._scene_error = f"Cannot build scene: {exc}"
        if self._view is not None:
            self._view.set_message(self._scene_error)
        self._touch(footer=True)

    def _render_frame(self, width_px: int, height_px: int):
        if self._scene is None:
            return None
        return render_frame(
            self._scene,
            width_px,
            height_px,
            yaw=self._yaw,
            pitch=self._pitch,
            zoom=self._zoom,
            show_bonds=self._show_bonds,
            show_cell=self._show_cell,
        )

    def _touch(self, *, footer: bool = False) -> None:
        self._refresh_info()
        if footer:
            self.refresh_footer()
        self._schedule_redraw()

    def _schedule_redraw(self) -> None:
        loop = self._state.loop
        if loop is None:
            # No loop to pace against (headless/tests): repaint immediately.
            if self._view is not None:
                self._view.mark_dirty()
            return
        if self._spinning or self._redraw_handle is not None:
            return
        render_seconds = self._view.last_render_seconds if self._view else 0.0
        self._redraw_handle = loop.set_alarm_in(
            _next_delay(REDRAW_INTERVAL, render_seconds), self._on_redraw
        )

    def _on_redraw(self, loop, _data=None) -> None:
        self._redraw_handle = None
        if self._router.current is not self:
            return
        if self._view is not None:
            self._view.mark_dirty()
        self._refresh_info()
        loop.draw_screen()

    def _cancel_redraw(self) -> None:
        loop = self._state.loop
        if self._redraw_handle is not None and loop is not None:
            try:
                loop.remove_alarm(self._redraw_handle)
            except Exception:
                pass
        self._redraw_handle = None

    # ------------------------------------------------------------------ #
    #  Header
    # ------------------------------------------------------------------ #

    def _spacegroup(self) -> str:
        if self._spg is None:
            result = self._result
            try:
                self._spg = (result.final_spg or result.initial_spg) if result else ""
            except Exception:
                self._spg = "N/A"
        return self._spg

    def _refresh_info(self) -> None:
        result = self._result
        if result is None:
            self._identity.set_text("  No structure to display")
            self._viewstate.set_text("")
            return

        scene = self._scene
        parts = [f"[{self._index + 1}/{len(self._results)}]"]
        if result.formula:
            parts.append(result.formula)
        if scene is not None and scene.n_atoms:
            parts.append(f"{scene.n_atoms} atoms")
        spg = self._spacegroup()
        if spg:
            parts.append(spg)
        try:
            parts.append(f"{result.display_epa:.4f} eV/at")
        except (TypeError, ValueError):
            pass
        parts.append(result.short_id)
        if scene is not None and scene.bonds_capped:
            parts.append("bonds skipped (too many atoms)")
        self._identity.set_text(" " + SEP.join(parts))

        state = [style_label(self._style)]
        if self._supercell > 1:
            state.append(f"{self._supercell}x{self._supercell}x{self._supercell}")
        state.append("bonds" if self._show_bonds else "no bonds")
        if scene is not None and scene.has_cell:
            state.append("cell" if self._show_cell else "no cell")
        state.append(
            f"yaw {round(math.degrees(self._yaw)) % 360}"
            f" pitch {_signed_degrees(self._pitch)}"
            f" zoom {self._zoom:.2f}x"
        )
        state.append(color_depth_label(self._state.color_depth))
        self._viewstate.set_text([("details", " " + SEP.join(state))])

    # ------------------------------------------------------------------ #
    #  Key bindings
    # ------------------------------------------------------------------ #

    def keypress(self, size: tuple, key: str) -> str | None:
        # Remember the key so one Rotate binding can serve all eight arrows.
        self._last_key = key if isinstance(key, str) else ""
        return super().keypress(size, key)

    def bindings(self) -> list[KeyBinding]:
        if not self._renderable:
            return []
        scene = self._scene
        return [
            KeyBinding(
                ("d", "n", "page down"), "Next", lambda: self._step(1),
                help="Next structure",
                enabled=lambda: self._index < len(self._results) - 1,
                priority=5,
            ),
            KeyBinding(
                ("a", "p", "page up"), "Prev", lambda: self._step(-1),
                help="Previous structure",
                enabled=lambda: self._index > 0,
                priority=6,
            ),
            KeyBinding(
                ("left", "right", "up", "down", "h", "j", "k", "l"),
                "Rotate", self._action_rotate,
                help="Rotate (arrows or h/j/k/l)", priority=10,
            ),
            KeyBinding(
                (" ",), lambda: "Stop" if self._spinning else "Spin",
                self._action_toggle_spin,
                help="Start or stop auto-rotation",
                enabled=lambda: self._state.loop is not None,
                priority=15,
            ),
            KeyBinding(
                ("f",), "Style", self._action_cycle_style,
                help="Cycle through ball-and-stick / spacefill / wireframe", priority=20,
            ),
            KeyBinding(
                ("b",), "Bonds", self._action_toggle_bonds,
                help="Show or hide bonds",
                enabled=lambda: not (scene is not None and scene.bonds_capped),
                priority=25,
            ),
            KeyBinding(
                ("c",), "Cell", self._action_toggle_cell,
                help="Toggle the unit-cell box",
                enabled=lambda: scene is not None and scene.has_cell,
                priority=30,
            ),
            KeyBinding(
                ("s",), lambda: "Cell x1" if self._supercell > 1 else "Cell x2",
                self._action_toggle_supercell,
                help="Toggle a 2x2x2 supercell",
                enabled=lambda: scene is not None and scene.can_supercell,
                priority=35,
            ),
            KeyBinding(
                ("+", "="), "Zoom+", lambda: self._zoom_by(ZOOM_STEP),
                help="Zoom in", priority=40,
            ),
            KeyBinding(
                ("-", "_"), "Zoom-", lambda: self._zoom_by(1.0 / ZOOM_STEP),
                help="Zoom out", priority=45,
            ),
            KeyBinding(
                ("r",), "Reset", self._action_reset,
                help="Reset rotation and zoom", priority=50,
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Actions
    # ------------------------------------------------------------------ #

    def _step(self, delta: int) -> None:
        if not self._results:
            return
        new_index = max(0, min(len(self._results) - 1, self._index + delta))
        if new_index == self._index:
            return
        self._index = new_index
        self._spg = None
        self._rebuild_scene()
        self._router.refresh_breadcrumb()
        if self._on_focus is not None:
            self._on_focus(self._results[new_index])

    def _action_rotate(self) -> None:
        delta = _ROTATE.get(self._last_key.lower())
        if delta is None:
            return
        
        self._yaw += delta[0]
        self._pitch += delta[1]
        self._touch()

    def _zoom_by(self, factor: float) -> None:
        self._zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom * factor))
        self._touch()

    def _action_reset(self) -> None:
        self._yaw, self._pitch, self._zoom = DEFAULT_YAW, DEFAULT_PITCH, 1.0
        self._touch()

    def _action_cycle_style(self) -> None:
        self._style = next_style(self._style)
        self._rebuild_scene()

    def _action_toggle_bonds(self) -> None:
        self._show_bonds = not self._show_bonds
        self._rebuild_scene()

    def _action_toggle_cell(self) -> None:
        self._show_cell = not self._show_cell
        self._touch()

    def _action_toggle_supercell(self) -> None:
        self._supercell = 1 if self._supercell > 1 else 2
        self._rebuild_scene()

    # ------------------------------------------------------------------ #
    #  Spin
    # ------------------------------------------------------------------ #

    def _action_toggle_spin(self) -> None:
        self._spinning = not self._spinning
        if self._spinning:
            self._start_spin()
        else:
            self._stop_spin()
        self._touch(footer=True)

    def _start_spin(self) -> None:
        loop = self._state.loop
        if loop is None or self._spin_handle is not None:
            return
        self._cancel_redraw()
        self._spin_last = time.monotonic()
        self._spin_handle = loop.set_alarm_in(SPIN_INTERVAL, self._on_spin_tick)

    def _stop_spin(self) -> None:
        loop = self._state.loop
        if self._spin_handle is not None and loop is not None:
            try:
                loop.remove_alarm(self._spin_handle)
            except Exception:
                pass
        self._spin_handle = None

    def _on_spin_tick(self, loop, _data=None) -> None:
        self._spin_handle = None
        if not self._spinning or self._router.current is not self:
            return
        now = time.monotonic()
        self._yaw += SPIN_RATE * (now - self._spin_last)
        self._spin_last = now
        if self._view is not None:
            self._view.mark_dirty()
        self._refresh_info()
        loop.draw_screen()

        render_seconds = self._view.last_render_seconds if self._view else 0.0
        self._spin_handle = loop.set_alarm_in(
            _next_delay(SPIN_INTERVAL, render_seconds), self._on_spin_tick
        )

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def on_resume(self) -> None:
        super().on_resume()
        if self._spinning:
            self._start_spin()

    def on_leave(self) -> None:
        self._stop_spin()
        self._cancel_redraw()
        super().on_leave()
