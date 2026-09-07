import urwid

from pathlib import Path

from rapmat.db_config import _DEFAULT_SQLITE_PATH
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.widgets.dialog import ModalDialog
from rapmat.utils.common import format_bytes


class DbSettingsScreen(ScreenBase):
    title = "DB Settings"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None
        self._status_text: urwid.Text | None = None
        self._path_edit: urwid.Edit | None = None
        self._usage_text: urwid.Text | None = None

    def build(self) -> urwid.Widget:
        self._main_body = self._build_body()
        self._frame = urwid.Frame(body=self._main_body)
        self.refresh_footer()
        return self._frame

    def extra_hints(self) -> list:
        return [("Tab", "Navigate", 10), ("Enter", "Apply", 20)]

    def esc_label(self) -> str:
        return "Cancel"

    def _dialog_host_get(self) -> "urwid.Widget | None":
        return self._frame.body if self._frame is not None else None

    def _dialog_host_set(self, widget: urwid.Widget) -> None:
        if self._frame is not None:
            self._frame.body = widget

    def _effective_path(self) -> str:
        from rapmat.db_config import load_db_config

        custom = load_db_config().get("general", {}).get("db_path", "")
        return custom or _DEFAULT_SQLITE_PATH

    def _usage_markup(self) -> list:
        stats = self._state.store.storage_stats()
        if stats is None:
            return [("details", "  Not available for this backend.")]

        pct = stats.reclaimable_fraction * 100.0
        style = "unconv" if pct >= 25.0 else "details"
        return [
            ("form_label", "  Size:        "),
            ("details", f"{format_bytes(stats.total_bytes)}\n"),
            ("form_label", "  Reclaimable: "),
            (style, f"{format_bytes(stats.reclaimable_bytes)} ({pct:.0f}%)"),
        ]

    def _refresh_usage(self) -> None:
        if self._usage_text is None:
            return
        try:
            self._usage_text.set_text(self._usage_markup())
        except Exception as exc:
            self._usage_text.set_text(("error", f"  Unavailable: {exc}"))

    def _build_body(self) -> urwid.Widget:
        from rapmat.db_config import _DB_CONFIG_FILE

        current_info = urwid.Text(
            [
                ("section", " Active Backend\n"),
                ("form_label", "  Mode:  "),
                ("details", "SQLite\n"),
                ("form_label", "  DB:    "),
                ("details", (self._state.store.get_url() or "(in-memory)") + "\n"),
                ("form_label", "  File:  "),
                ("details", str(_DB_CONFIG_FILE)),
            ]
        )

        self._path_edit = urwid.Edit(edit_text=self._effective_path())
        path_row = urwid.Columns(
            [
                (14, urwid.Text(("form_label", "Data path:"), align="right")),
                (
                    "weight",
                    1,
                    urwid.AttrMap(self._path_edit, None, focus_map="focus"),
                ),
            ],
            dividechars=1,
        )

        self._usage_text = urwid.Text("")
        self._refresh_usage()

        compact_btn = urwid.AttrMap(
            urwid.Button("Compact Now", on_press=self._on_compact),
            None,
            focus_map="btn_focus",
        )

        self._status_text = urwid.Text("")

        test_btn = urwid.AttrMap(
            urwid.Button("Test", on_press=self._on_test),
            None,
            focus_map="btn_focus",
        )
        save_btn = urwid.AttrMap(
            urwid.Button("Save", on_press=self._on_save),
            None,
            focus_map="btn_focus",
        )
        clear_btn = urwid.AttrMap(
            urwid.Button("Clear Config", on_press=self._on_clear),
            None,
            focus_map="btn_focus",
        )
        btn_row = urwid.Columns(
            [
                ("weight", 1, test_btn),
                ("weight", 1, save_btn),
                ("weight", 1, clear_btn),
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
                        ("details", "  Leave as is to use the default location.")
                    ),
                ),
                ("pack", urwid.Divider()),
                ("pack", urwid.Text(("section", " Disk Usage"))),
                ("pack", self._usage_text),
                ("pack", compact_btn),
                (
                    "pack",
                    urwid.Text(
                        (
                            "details",
                            "  The file will be rebuilt to release deleted rows."
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

    def _on_test(self, _btn) -> None:
        if self._status_text is None:
            return
        path = self._edited_path() or _DEFAULT_SQLITE_PATH
        self._status_text.set_text(("details", "  Testing SQLite..."))
        try:
            from rapmat.storage.sqlite_store import SQLiteStore

            store = SQLiteStore.from_path(Path(path))
            store.close()
            self._status_text.set_text(("success", "  SQLite OK."))
        except Exception as exc:
            self._status_text.set_text(("error", f"  SQLite failed: {exc}"))

    def _on_save(self, _btn) -> None:
        if self._status_text is None:
            return
        try:
            from rapmat.db_config import resolve_store, save_db_config

            path = self._edited_path()
            db_path = "" if path == _DEFAULT_SQLITE_PATH else path
            save_db_config(general={"mode": "sqlite", "db_path": db_path})

            new_store = resolve_store()
            self._state.reconnect(new_store)
            self._status_text.set_text(("success", "  Saved & reconnected."))
        except Exception as exc:
            self._status_text.set_text(("error", f"  Save failed: {exc}"))

    def _on_compact(self, _btn) -> None:
        if self._status_text is None:
            return
        if self.has_live_tasks():
            self._status_text.set_text(("unconv", "  Already compacting."))
            return

        stats = self._state.store.storage_stats()
        if stats is None:
            self._status_text.set_text(
                ("unconv", "  Not supported by this backend.")
            )
            return
        if stats.free_pages == 0:
            self._status_text.set_text(("success", "  Nothing to reclaim."))
            return

        self.confirm_dialog(
            "Compact Database",
            f"Rebuild the database to release "
            f"{format_bytes(stats.reclaimable_bytes)}?\n\n"
            f"Needs up to {format_bytes(stats.total_bytes)} of free disk space\n"
            f"while it runs. Do not interrupt rapmat.",
            lambda: self._start_compact(stats.total_bytes),
        )

    def _start_compact(self, before_bytes: int) -> None:
        if self._status_text is None:
            return
        self._status_text.set_text(("details", "  Compacting, please wait..."))
        if self._state.loop is not None:
            self._state.loop.draw_screen()

        def _work(progress) -> None:
            progress.log("Running VACUUM")
            self._state.store.vacuum()

        self.run_task(
            _work,
            on_complete=lambda: self._on_compact_done(before_bytes),
            on_error=self._on_compact_error,
        )

    def _on_compact_done(self, before_bytes: int) -> None:
        self._refresh_usage()
        after = self._state.store.storage_stats()
        if self._status_text is None:
            return
        if after is None:
            self._status_text.set_text(("success", "  Compacted."))
            return
        saved = max(before_bytes - after.total_bytes, 0)
        self._status_text.set_text(
            (
                "success",
                f"  Compacted: {format_bytes(before_bytes)} -> "
                f"{format_bytes(after.total_bytes)} "
                f"(freed {format_bytes(saved)}).",
            )
        )
        self._refresh_footer_if_current()

    def _on_compact_error(self, error: str) -> None:
        if self._status_text is not None:
            self._status_text.set_text(("error", f"  Compact failed: {error}"))
        self._refresh_footer_if_current()

    def _refresh_footer_if_current(self) -> None:
        if self._router.current is self:
            self.refresh_footer()

    def _on_clear(self, _btn) -> None:
        if self._frame is None or self._main_body is None:
            return
        dlg = ModalDialog.confirm(
            "Clear Configuration",
            "Delete the saved DB configuration and reset to defaults?",
            parent=self._main_body,
            on_close=self._on_clear_confirm,
        )
        self._frame.body = dlg

    def _on_clear_confirm(self, confirmed: bool) -> None:
        if self._frame is None or self._main_body is None:
            return
        self._frame.body = self._main_body
        if confirmed:
            from rapmat.db_config import clear_db_config

            removed = clear_db_config()
            if self._status_text:
                if removed:
                    self._status_text.set_text(("success", "  Configuration cleared."))
                else:
                    self._status_text.set_text(
                        ("unconv", "  No configuration to clear.")
                    )

