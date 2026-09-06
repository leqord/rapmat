from pathlib import Path

import urwid

from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState

_TEMP_NOTE = "(temporary)"


class CalcSettingsScreen(ScreenBase):
    title = "Calc Settings"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._status_text: urwid.Text | None = None
        self._path_edit: urwid.Edit | None = None

    def build(self) -> urwid.Widget:
        return self._build_body()

    def extra_hints(self) -> list:
        return [("Tab", "Navigate", 10), ("Enter", "Apply", 20)]

    def esc_label(self) -> str:
        return "Back"

    def _build_body(self) -> urwid.Widget:
        from rapmat.app_config import resolve_calc_root, settings_file_path

        current = resolve_calc_root()

        current_info = urwid.Text(
            [
                ("section", " Calculation Directories\n"),
                ("form_label", "  Root:  "),
                ("details", (current or _TEMP_NOTE) + "\n"),
                ("form_label", "  File:  "),
                ("details", str(settings_file_path())),
            ]
        )

        self._path_edit = urwid.Edit(edit_text=current)
        path_row = urwid.Columns(
            [
                (14, urwid.Text(("form_label", "Workdirs root:"), align="right")),
                (
                    "weight",
                    1,
                    urwid.AttrMap(self._path_edit, None, focus_map="focus"),
                ),
            ],
            dividechars=1,
        )

        self._status_text = urwid.Text("")

        def _button(label: str, callback) -> urwid.Widget:
            return urwid.AttrMap(
                urwid.Button(label, on_press=callback), None, focus_map="btn_focus"
            )

        btn_row = urwid.Columns(
            [
                ("weight", 1, _button("Test", self._on_test)),
                ("weight", 1, _button("Save", self._on_save)),
                ("weight", 1, _button("Clear", self._on_clear)),
            ],
            dividechars=2,
        )

        body_pile = urwid.Pile(
            [
                ("pack", current_info),
                ("pack", urwid.Divider()),
                ("pack", urwid.Text(("section", " Storage Location"))),
                ("pack", path_row),
                (
                    "pack",
                    urwid.Text(
                        (
                            "details",
                            "  Leave blank to use a temporary directory.",
                        )
                    ),
                ),
                ("pack", urwid.Divider()),
                ("pack", self._status_text),
                ("pack", urwid.Divider()),
                ("pack", btn_row),
            ]
        )

        listbox = urwid.ListBox(urwid.SimpleListWalker([body_pile]))
        scrollable = urwid.ScrollBar(
            listbox,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )
        return urwid.Padding(scrollable, left=2, right=2)

    def _edited_path(self) -> str:
        if self._path_edit is None:
            return ""
        return self._path_edit.get_edit_text().strip()

    def _set_status(self, style: str, message: str) -> None:
        if self._status_text is not None:
            self._status_text.set_text((style, f"  {message}"))

    def _on_test(self, _btn) -> None:
        path = self._edited_path()
        if not path:
            self._set_status("details", f"Blank: {_TEMP_NOTE}")
            return

        probe = Path(path).expanduser() / ".rapmat-write-test"
        try:
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            self._set_status("success", "Writable.")
        except OSError as exc:
            self._set_status("error", f"Not usable: {exc}")

    def _on_save(self, _btn) -> None:
        from rapmat.app_config import persist_calc_root

        path = self._edited_path()
        try:
            changed = persist_calc_root(path)
        except OSError as exc:
            self._set_status("error", f"Save failed: {exc}")
            return

        if not changed:
            self._set_status("unconv", "No change.")
        elif path:
            self._set_status("success", f"Calculations will be kept in {path}")
        else:
            self._set_status("success", f"Cleared {_TEMP_NOTE}")

    def _on_clear(self, _btn) -> None:
        if self._path_edit is not None:
            self._path_edit.set_edit_text("")
        self._on_save(_btn)
