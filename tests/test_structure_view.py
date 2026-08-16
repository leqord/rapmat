"""Structure viewer."""

import math
import sys

import numpy as np
import pytest

if sys.platform == "win32":
    sys.modules.pop("urwid.display.curses", None)

from rapmat.storage.sqlite_store import SQLiteStore
from rapmat.tui.app import RapmatApp
from rapmat.tui.screens.structure_view import (
    DEFAULT_PITCH,
    DEFAULT_YAW,
    ZOOM_MAX,
    ZOOM_MIN,
    StructureViewScreen,
)
from rapmat.tui.state import AppState

_SIZE = (121, 38)


def _rows(n: int = 3):
    from ase.build import bulk

    from rapmat.core.entities import ResultRow, Structure

    rows = []
    for idx in range(n):
        struct = Structure(
            id=f"smoke-run/{idx + 1}",
            status="relaxed",
            energy_per_atom=-5.0 + idx * 0.1,
            converged=True,
            final_atoms=bulk("NaCl", "rocksalt", a=5.6 + idx * 0.1, cubic=True),
        )
        rows.append(ResultRow(structure=struct, index=idx + 1, run_name="smoke-run"))
    return rows


def _molecule_row():
    from ase import Atoms

    from rapmat.core.entities import ResultRow, Structure

    water = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])
    struct = Structure(
        id="mol/1", status="relaxed", energy_per_atom=-1.0,
        converged=True, final_atoms=water,
    )
    return ResultRow(structure=struct, index=1, run_name="smoke-run")


class _FakeLoop:
    def __init__(self):
        self.alarms = []
        self.removed = []
        self.drawn = 0

    def set_alarm_in(self, delay, callback, *args):
        handle = (delay, callback, len(self.alarms))
        self.alarms.append(handle)
        return handle

    def remove_alarm(self, handle):
        self.removed.append(handle)
        return True

    def draw_screen(self):
        self.drawn += 1


@pytest.fixture
def app_env():
    store = SQLiteStore(":memory:")
    state = AppState(store=store)
    app = RapmatApp(state)
    state.loop = None
    yield state, app
    store.close()


@pytest.fixture
def view_env(app_env):
    state, app = app_env
    screen, widget = _make(state, app, _rows(), 0)
    return state, screen, widget


def _make(state, app, rows, index=0, **kw):
    screen = StructureViewScreen(state, app._router, rows, index, **kw)
    app._router.push(screen)
    widget = app._router._stack[-1][1]
    widget.render(_SIZE, focus=True)
    return screen, widget


def _binding(screen, key):
    return next(b for b in screen.bindings() if key in b.keys)


def _canvas_text(canvas) -> str:
    return "\n".join(
        b"".join(part[2] for part in row).decode("utf-8", "replace")
        for row in canvas.content()
    )


# ------------------------------------------------------------------ #
#  Camera
# ------------------------------------------------------------------ #


def test_rotate_keys_change_the_camera(view_env):
    _state, screen, widget = view_env
    yaw, pitch = screen._yaw, screen._pitch

    screen.keypress((), "right")
    assert screen._yaw > yaw
    screen.keypress((), "h")
    assert screen._yaw == pytest.approx(yaw)

    screen.keypress((), "up")
    assert screen._pitch < pitch
    screen.keypress((), "j")
    assert screen._pitch == pytest.approx(pitch)
    widget.render(_SIZE, focus=True)


def test_arrow_and_vim_keys_agree(view_env):
    _state, screen, _widget = view_env
    screen.keypress((), "right")
    by_arrow = screen._yaw
    screen._action_reset()
    screen.keypress((), "l")
    assert screen._yaw == pytest.approx(by_arrow)


def test_pitch_wraps_instead_of_stopping(view_env):
    _state, screen, widget = view_env
    start = screen._pitch
    for _ in range(60):                       # 60 * 6deg = one full turn
        screen.keypress((), "down")
    assert screen._pitch == pytest.approx(start + 2 * math.pi)
    widget.render(_SIZE, focus=True)

    for _ in range(30):
        screen.keypress((), "up")
    assert screen._pitch == pytest.approx(start + math.pi)


def test_upside_down_view_still_renders(view_env):
    from rapmat.tui.widgets.halfblock import GLYPH

    _state, screen, widget = view_env
    for _ in range(30):                       # past the pole
        screen.keypress((), "down")
    text = _canvas_text(widget.render(_SIZE, focus=True))
    assert text.count(GLYPH) > 1000


def test_rotation_matrix_stays_valid_past_the_pole():
    from rapmat.core.render3d import rot_matrix

    for degrees in (0, 90, 135, 180, 270, 359):
        r = rot_matrix(0.5, math.radians(degrees))
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(r)) == pytest.approx(1.0)


def test_yaw_response_is_independent_of_pitch():
    from rapmat.core.render3d import rot_matrix

    v = np.array([1.0, 0.4, 0.0])
    step = math.radians(6)
    deltas = [
        ((rot_matrix(0.5 + step, p) @ v)[0] - (rot_matrix(0.5, p) @ v)[0])
        for p in np.radians([0, 60, 120, 180, 240, 300])
    ]
    assert all(d == pytest.approx(deltas[0]) for d in deltas)


@pytest.mark.parametrize(
    "degrees, shown",
    [(0, 0), (-18, -18), (90, 90), (180, 180), (181, -179), (270, -90), (342, -18)],
)
def test_pitch_is_displayed_as_a_signed_tilt(degrees, shown):
    from rapmat.tui.screens.structure_view import _signed_degrees

    assert _signed_degrees(math.radians(degrees)) == shown


def test_zoom_is_clamped(view_env):
    _state, screen, _widget = view_env
    for _ in range(100):
        screen.keypress((), "+")
    assert screen._zoom == pytest.approx(ZOOM_MAX)
    for _ in range(200):
        screen.keypress((), "-")
    assert screen._zoom == pytest.approx(ZOOM_MIN)


def test_reset_restores_the_camera_but_not_the_style(view_env):
    _state, screen, _widget = view_env
    screen.keypress((), "right")
    screen.keypress((), "+")
    screen.keypress((), "f")
    screen.keypress((), "r")
    assert screen._yaw == pytest.approx(DEFAULT_YAW)
    assert screen._pitch == pytest.approx(DEFAULT_PITCH)
    assert screen._zoom == pytest.approx(1.0)
    assert screen._style != "ball"


# ------------------------------------------------------------------ #
#  Scene toggles
# ------------------------------------------------------------------ #


def test_style_cycles_back_around(view_env):
    _state, screen, widget = view_env
    assert screen._style == "ball"
    for expected in ("spacefill", "wireframe", "ball"):
        screen.keypress((), "f")
        assert screen._style == expected
        widget.render(_SIZE, focus=True)


def test_bonds_and_cell_toggle(view_env):
    _state, screen, widget = view_env
    assert screen._scene.n_bonds > 0
    screen.keypress((), "b")
    assert screen._show_bonds is False
    assert screen._scene.n_bonds == 0
    screen.keypress((), "c")
    assert screen._show_cell is False
    widget.render(_SIZE, focus=True)


def test_supercell_toggle(view_env):
    _state, screen, widget = view_env
    base = screen._scene.n_atoms
    screen.keypress((), "s")
    assert screen._supercell == 2
    assert screen._scene.n_atoms == base * 8
    widget.render(_SIZE, focus=True)
    screen.keypress((), "s")
    assert screen._supercell == 1
    assert screen._scene.n_atoms == base


def test_cell_bindings_disabled_for_a_molecule(app_env):
    state, app = app_env
    screen, _widget = _make(state, app, [_molecule_row()])
    assert screen._scene.has_cell is False
    assert _binding(screen, "s").is_enabled() is False
    assert _binding(screen, "c").is_enabled() is False
    screen.keypress((), "s")
    assert screen._supercell == 1


# ------------------------------------------------------------------ #
#  Browsing
# ------------------------------------------------------------------ #


def test_next_and_prev_step_through_the_list(view_env):
    _state, screen, widget = view_env
    assert screen._index == 0
    screen.keypress((), "d")
    assert screen._index == 1
    widget.render(_SIZE, focus=True)
    screen.keypress((), "a")
    assert screen._index == 0


def test_stepping_clamps_at_both_ends(view_env):
    _state, screen, _widget = view_env
    assert _binding(screen, "a").is_enabled() is False
    screen.keypress((), "a")
    assert screen._index == 0

    for _ in range(10):
        screen.keypress((), "d")
    assert screen._index == len(screen._results) - 1
    assert _binding(screen, "d").is_enabled() is False


def test_page_keys_step_too(view_env):
    _state, screen, _widget = view_env
    screen.keypress((), "page down")
    assert screen._index == 1
    screen.keypress((), "page up")
    assert screen._index == 0


@pytest.mark.parametrize("fwd, back", [("d", "a"), ("n", "p"), ("page down", "page up")])
def test_every_step_alias_works(view_env, fwd, back):
    _state, screen, _widget = view_env
    screen.keypress((), fwd)
    assert screen._index == 1
    screen.keypress((), back)
    assert screen._index == 0


def test_step_keys_are_advertised_as_a_and_d(view_env):
    _state, screen, _widget = view_env
    assert _binding(screen, "d").key_display() == "d"
    assert _binding(screen, "a").key_display() == "a"


def test_step_keys_do_not_collide_with_view_keys(view_env):
    _state, screen, _widget = view_env
    owners = {k: b.label_text() for b in screen.bindings() for k in b.keys}
    assert owners["a"] == "Prev"
    assert owners["d"] == "Next"


def test_step_reports_the_new_row(app_env):
    state, app = app_env
    seen = []
    rows = _rows()
    screen, _widget = _make(state, app, rows, 0, on_focus=seen.append)
    screen.keypress((), "d")
    assert seen == [rows[1]]


def test_step_keeps_the_camera(view_env):
    _state, screen, _widget = view_env
    screen.keypress((), "right")
    screen.keypress((), "+")
    screen.keypress((), "f")
    yaw, zoom, style = screen._yaw, screen._zoom, screen._style
    before = screen._scene

    screen.keypress((), "d")

    assert screen._yaw == pytest.approx(yaw)
    assert screen._zoom == pytest.approx(zoom)
    assert screen._style == style
    assert screen._scene is not before


def test_step_updates_the_header_and_breadcrumb(view_env):
    _state, screen, widget = view_env
    first = screen.breadcrumb_title
    assert "[1/3]" in screen._identity.text
    screen.keypress((), "d")
    widget.render(_SIZE, focus=True)
    assert screen.breadcrumb_title != first
    assert "[2/3]" in screen._identity.text


def test_single_entry_list_cannot_step(app_env):
    state, app = app_env
    screen, _widget = _make(state, app, _rows(1))
    assert _binding(screen, "d").is_enabled() is False
    assert _binding(screen, "a").is_enabled() is False
    screen.keypress((), "d")
    assert screen._index == 0


def test_out_of_range_start_index_is_clamped(app_env):
    state, app = app_env
    screen = StructureViewScreen(state, app._router, _rows(), 99)
    assert screen._index == 2


def test_redraw_does_not_recompute_the_spacegroup(view_env, monkeypatch):
    import rapmat.storage.models as models

    calls = []
    real = models.format_spg

    def counted(atoms, symprec=1e-3):
        calls.append(atoms)
        return real(atoms, symprec=symprec)

    monkeypatch.setattr(models, "format_spg", counted)

    _state, screen, widget = view_env
    screen._spg = None
    screen._refresh_info()
    baseline = len(calls)
    assert baseline > 0

    for _ in range(20):
        screen.keypress((), "right")
    widget.render(_SIZE, focus=True)
    assert len(calls) == baseline

    screen.keypress((), "d")
    assert len(calls) > baseline


# ------------------------------------------------------------------ #
#  Degraded terminals and edge cases
# ------------------------------------------------------------------ #


def test_colour_mode_is_always_shown(app_env):
    from rapmat.tui.theme import TRUECOLOR

    state, app = app_env
    state.color_depth = TRUECOLOR
    screen, _widget = _make(state, app, _rows())
    assert "truecolor" in screen._viewstate.text


def test_degraded_colour_mode_says_it_is_approximated(app_env):
    state, app = app_env
    state.color_depth = 256
    screen, _widget = _make(state, app, _rows())
    assert "256 colours (approximated)" in screen._viewstate.text


@pytest.mark.parametrize(
    "depth, shown",
    [(2 ** 24, "truecolor"), (256, "256 colours (approximated)"), (16, "16 colours")],
)
def test_colour_depth_label(depth, shown):
    from rapmat.tui.theme import color_depth_label

    assert color_depth_label(depth) == shown


def test_sixteen_colours_explains_itself(app_env):
    state, app = app_env
    state.color_depth = 16
    screen, widget = _make(state, app, _rows())
    assert "16 colours" in _canvas_text(widget.render(_SIZE, focus=True))
    assert screen.bindings() == []


def test_non_utf8_encoding_explains_itself(app_env, monkeypatch):
    import rapmat.tui.screens.structure_view as sv

    monkeypatch.setattr(sv, "glyph_is_renderable", lambda: False)
    state, app = app_env
    screen, widget = _make(state, app, _rows())
    assert "UTF-8" in _canvas_text(widget.render(_SIZE, focus=True))
    assert screen.bindings() == []


def test_empty_list_renders_and_binds_nothing(app_env):
    state, app = app_env
    screen, _widget = _make(state, app, [])
    assert screen.bindings() == []
    assert screen.breadcrumb_title == "3D View"


def test_scene_failure_is_reported_not_raised(app_env, monkeypatch):
    import rapmat.tui.screens.structure_view as sv

    def boom(*_a, **_kw):
        raise RuntimeError("bad cell")

    monkeypatch.setattr(sv, "build_scene", boom)
    state, app = app_env
    screen, widget = _make(state, app, _rows())
    assert "Cannot build scene" in _canvas_text(widget.render(_SIZE, focus=True))
    assert screen.bindings() == []


@pytest.mark.parametrize("size", [(121, 13), (80, 24), (20, 6), (10, 3), (4, 2)])
def test_renders_at_extreme_sizes(view_env, size):
    _state, _screen, widget = view_env
    canvas = widget.render(size, focus=True)
    assert (canvas.cols(), canvas.rows()) == size


def test_the_picture_is_actually_drawn(view_env):
    from rapmat.tui.widgets.halfblock import GLYPH

    _state, _screen, widget = view_env
    text = _canvas_text(widget.render(_SIZE, focus=True))
    assert text.count(GLYPH) > 1000


# ------------------------------------------------------------------ #
#  Spin
# ------------------------------------------------------------------ #


def test_spin_schedules_and_on_leave_cancels(view_env):
    _state, screen, _widget = view_env
    loop = _FakeLoop()
    screen._state.loop = loop

    screen.keypress((), " ")
    assert screen._spinning is True
    assert screen._spin_handle is not None
    assert loop.alarms

    screen.on_leave()
    assert screen._spin_handle is None
    assert loop.removed


def test_spin_toggles_back_off(view_env):
    _state, screen, _widget = view_env
    screen._state.loop = _FakeLoop()
    screen.keypress((), " ")
    screen.keypress((), " ")
    assert screen._spinning is False
    assert screen._spin_handle is None


def test_spin_tick_advances_yaw_and_rearms(view_env):
    _state, screen, _widget = view_env
    loop = _FakeLoop()
    screen._state.loop = loop
    screen.keypress((), " ")
    yaw = screen._yaw

    screen._spin_last -= 0.5
    screen._on_spin_tick(loop)

    assert screen._yaw > yaw
    assert screen._spin_handle is not None
    assert loop.drawn >= 1


def test_spin_tick_stops_once_the_screen_is_covered(view_env, app_env):
    _state, screen, _widget = view_env
    state, app = app_env
    loop = _FakeLoop()
    screen._state.loop = loop
    screen.keypress((), " ")

    app._router.push(StructureViewScreen(state, app._router, _rows(), 0))
    assert app._router.current is not screen

    screen._spin_handle = None
    screen._on_spin_tick(loop)
    assert screen._spin_handle is None


def test_spin_without_a_loop_is_inert(view_env):
    _state, screen, _widget = view_env
    screen._state.loop = None
    assert _binding(screen, " ").is_enabled() is False
    screen._action_toggle_spin()
    assert screen._spin_handle is None


# ------------------------------------------------------------------ #
#  Redraw pacing
# ------------------------------------------------------------------ #


def test_rotation_does_not_rebuild_the_footer(view_env):
    _state, screen, _widget = view_env
    calls = []
    screen.refresh_footer = lambda message="": calls.append(message)

    for _ in range(10):
        screen.keypress((), "right")
    screen.keypress((), "+")
    screen.keypress((), "r")
    assert calls == []


def test_scene_changes_do_rebuild_the_footer(view_env):
    _state, screen, _widget = view_env
    for key in ("f", "b", "s", "d"):
        calls = []
        screen.refresh_footer = lambda message="": calls.append(message)
        screen.keypress((), key)
        assert calls, key
        del screen.refresh_footer


def test_cell_toggle_is_camera_only(view_env):
    _state, screen, _widget = view_env
    before = screen._scene
    calls = []
    screen.refresh_footer = lambda message="": calls.append(message)
    screen.keypress((), "c")
    assert calls == []
    assert screen._scene is before


def test_repeated_keys_coalesce_into_one_redraw(view_env):
    _state, screen, _widget = view_env
    loop = _FakeLoop()
    screen._state.loop = loop

    for _ in range(30):
        screen.keypress((), "right")

    assert len(loop.alarms) == 1
    assert screen._redraw_handle is not None


def test_redraw_alarm_paints_once_then_rearms_on_demand(view_env):
    _state, screen, _widget = view_env
    loop = _FakeLoop()
    screen._state.loop = loop

    screen.keypress((), "right")
    screen._on_redraw(loop)
    assert loop.drawn == 1
    assert screen._redraw_handle is None

    screen.keypress((), "right")
    assert len(loop.alarms) == 2


def test_redraw_backs_off_on_a_slow_terminal(view_env):
    from rapmat.tui.screens.structure_view import REDRAW_INTERVAL

    _state, screen, _widget = view_env
    loop = _FakeLoop()
    screen._state.loop = loop
    screen._view.last_render_seconds = 0.2

    screen.keypress((), "right")
    assert loop.alarms[0][0] == pytest.approx(0.3)
    assert loop.alarms[0][0] > REDRAW_INTERVAL


def test_stale_redraw_does_not_paint_a_covered_screen(view_env, app_env):
    _state, screen, _widget = view_env
    state, app = app_env
    loop = _FakeLoop()
    screen._state.loop = loop
    screen.keypress((), "right")

    app._router.push(StructureViewScreen(state, app._router, _rows(), 0))
    screen._on_redraw(loop)
    assert loop.drawn == 0


def test_spin_toggle_rebuilds_the_footer(view_env):
    _state, screen, _widget = view_env
    screen._state.loop = _FakeLoop()
    calls = []
    screen.refresh_footer = lambda message="": calls.append(message)
    screen.keypress((), " ")
    assert calls


def test_starting_spin_supersedes_a_pending_redraw(view_env):
    _state, screen, _widget = view_env
    loop = _FakeLoop()
    screen._state.loop = loop

    screen.keypress((), "right")
    pending = screen._redraw_handle
    screen.keypress((), " ")

    assert pending in loop.removed
    assert screen._redraw_handle is None
    assert screen._spin_handle is not None


def test_spinning_does_not_schedule_extra_redraws(view_env):
    _state, screen, _widget = view_env
    loop = _FakeLoop()
    screen._state.loop = loop
    screen.keypress((), " ")
    before = len(loop.alarms)

    for _ in range(10):
        screen.keypress((), "right")
    assert screen._redraw_handle is None
    assert len(loop.alarms) == before


def test_on_leave_cancels_a_pending_redraw(view_env):
    _state, screen, _widget = view_env
    loop = _FakeLoop()
    screen._state.loop = loop
    screen.keypress((), "right")

    screen.on_leave()
    assert screen._redraw_handle is None
    assert loop.removed


def test_without_a_loop_repaints_immediately(view_env):
    _state, screen, widget = view_env
    screen._state.loop = None
    screen.keypress((), "right")
    assert screen._redraw_handle is None
    widget.render(_SIZE, focus=True)


def test_on_resume_restarts_spin(view_env):
    _state, screen, _widget = view_env
    screen._state.loop = _FakeLoop()
    screen.keypress((), " ")
    screen.on_leave()
    assert screen._spin_handle is None
    screen.on_resume()
    assert screen._spin_handle is not None
