import sys

import urwid


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
        name, fg, bg = entry[0], entry[1], entry[2]
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


if __name__ == "__main__":
    test_render()
