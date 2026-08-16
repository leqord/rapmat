import sys

import pytest



def test_render():
    """Once in the past there was a crash."""
    from rapmat.storage import SQLiteStore
    from rapmat.tui.app import RapmatApp
    from rapmat.tui.screens.home import HomeScreen
    from rapmat.tui.state import AppState

    store = SQLiteStore(":memory:")
    state = AppState(store=store)
    app = RapmatApp(state)

    screen = HomeScreen(state, app._router)
    widget = screen.build()

    size = (121, 13)
    try:
        canv = widget.render(size, focus=True)
        print("Success! Canvas size:", canv.cols(), "x", canv.rows())
    except Exception as e:
        print("Crash:", e)
        sys.exit(1)


def test_dialog_remap_covers_default_bg_attrs():
    from rapmat.tui.theme import DIALOG_REMAP, PALETTE

    by_name = {entry[0]: entry for entry in PALETTE}

    for entry in PALETTE:
        name, bg = entry[0], entry[2]
        if name.endswith("@dlg"):
            continue
        if bg != "default":

            assert name not in DIALOG_REMAP
            continue
        assert name in DIALOG_REMAP, f"{name} has a default bg but is not remapped"
        twin = DIALOG_REMAP[name]
        twin_entry = by_name[twin]
        assert twin_entry[2] == "dark gray", f"{twin} should have a grey background"
        assert twin_entry[1] != twin_entry[2], f"{twin} foreground is invisible on grey"

        assert twin_entry[3:] == entry[3:]

    assert DIALOG_REMAP[None] == "dialog"


# ------------------------------------------------------------------ #
#  Colour depth
# ------------------------------------------------------------------ #


def test_truecolor_leaves_named_palette_escapes_unchanged():
    import urwid

    from rapmat.tui.theme import PALETTE, TRUECOLOR

    screen = urwid.display.raw.Screen()
    screen.register_palette(PALETTE)

    screen.set_terminal_properties(colors=16)
    before = dict(screen._pal_escape)

    screen.set_terminal_properties(colors=TRUECOLOR)
    after = dict(screen._pal_escape)

    assert set(before) == set(after)
    for name in before:
        assert before[name] == after[name], name
    assert before["header"] == "\x1b[0;97;44m"


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"COLORTERM": "truecolor"}, 2 ** 24),
        ({"COLORTERM": "24bit"}, 2 ** 24),
        ({"WT_SESSION": "abc"}, 2 ** 24),
        ({"KITTY_WINDOW_ID": "1"}, 2 ** 24),
        ({"TERM_PROGRAM": "vscode"}, 2 ** 24),
        ({"TERM": "xterm-direct"}, 2 ** 24),
        ({"TERM": "xterm-256color"}, 256),
        ({"TERM": "xterm"}, 16),
        ({"TERM": "dumb"}, 16),
        ({"COLORTERM": "truecolor", "RAPMAT_COLORS": "16"}, 16),
        ({"TERM": "xterm", "RAPMAT_COLORS": "truecolor"}, 2 ** 24),
        ({"TERM": "xterm", "RAPMAT_COLORS": "256"}, 256),
        ({"TERM": "xterm", "RAPMAT_COLORS": "nonsense"}, 16),
    ],
)
def test_detect_color_depth(env, expected):
    from rapmat.tui.theme import detect_color_depth

    assert detect_color_depth(env) == expected


def test_apply_color_depth_falls_back_when_the_setter_raises():
    from rapmat.tui.theme import apply_color_depth

    class _Screen:
        def __init__(self):
            self.calls = []

        def set_terminal_properties(self, colors=None):
            self.calls.append(colors)
            if colors == 2 ** 24:
                raise RuntimeError("no truecolor here")

    screen = _Screen()
    assert apply_color_depth(screen, 2 ** 24) == 256
    assert screen.calls == [2 ** 24, 256]


def test_apply_color_depth_never_upgrades_past_the_request():
    from rapmat.tui.theme import apply_color_depth

    class _Screen:
        def __init__(self):
            self.calls = []

        def set_terminal_properties(self, colors=None):
            self.calls.append(colors)

    screen = _Screen()
    assert apply_color_depth(screen, 16) == 16
    assert screen.calls == [16]


def test_apply_color_depth_without_a_setter():
    from rapmat.tui.theme import apply_color_depth

    assert apply_color_depth(object()) == 16


if __name__ == "__main__":
    test_render()
