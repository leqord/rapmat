from typing import Callable, Sequence

import urwid

from rapmat.tui.theme import DIALOG_REMAP


class ModalDialog(urwid.WidgetWrap):
    signals = ["close"]

    def __init__(
        self,
        title: str,
        body_widget: urwid.Widget,
        parent: urwid.Widget,
        *,
        width: int = 50,
        min_width: int = 40,
    ) -> None:
        self._parent = parent
        self._esc_handler: Callable[[], None] | None = None

        inner = urwid.LineBox(
            urwid.Padding(body_widget, left=1, right=1),
            title=title,
        )
        overlay = urwid.Overlay(
            urwid.AttrMap(inner, DIALOG_REMAP),
            parent,
            align=urwid.CENTER,
            width=(urwid.RELATIVE, width),
            valign=urwid.MIDDLE,
            height=urwid.PACK,
            min_width=min_width,
        )
        super().__init__(overlay)

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "esc" and self._esc_handler is not None:
            self._esc_handler()
            return None
        return super().keypress(size, key)

    # ------------------------------------------------------------------ #
    #  Static constructors
    # ------------------------------------------------------------------ #

    @staticmethod
    def confirm(
        title: str,
        message: str,
        parent: urwid.Widget,
        on_close: Callable[[bool], None],
    ) -> "ModalDialog":
        dlg: ModalDialog | None = None

        def _yes(btn: urwid.Button) -> None:
            assert dlg is not None
            dlg._emit("close", True)
            on_close(True)

        def _no(btn: urwid.Button) -> None:
            assert dlg is not None
            dlg._emit("close", False)
            on_close(False)

        yes_btn = urwid.AttrMap(
            urwid.Button("Yes", on_press=_yes), None, focus_map="btn_focus"
        )
        no_btn = urwid.AttrMap(
            urwid.Button("No", on_press=_no), None, focus_map="btn_focus"
        )

        body = urwid.Pile(
            [
                urwid.Text(message),
                urwid.Divider(),
                urwid.Columns(
                    [("weight", 1, yes_btn), ("weight", 1, no_btn)],
                    dividechars=2,
                ),
            ]
        )
        dlg = ModalDialog(title, body, parent)

        def _esc() -> None:
            dlg._emit("close", False)
            on_close(False)

        dlg._esc_handler = _esc
        return dlg

    @staticmethod
    def info(
        title: str,
        message: str,
        parent: urwid.Widget,
        on_close: Callable[[], None],
    ) -> "ModalDialog":
        dlg: ModalDialog | None = None

        def _ok(btn: urwid.Button) -> None:
            assert dlg is not None
            dlg._emit("close", True)
            on_close()

        ok_btn = urwid.AttrMap(
            urwid.Button("OK", on_press=_ok), None, focus_map="btn_focus"
        )

        body = urwid.Pile(
            [
                urwid.Text(message),
                urwid.Divider(),
                urwid.Padding(ok_btn, align=urwid.CENTER, width=10),
            ]
        )
        dlg = ModalDialog(title, body, parent)

        def _esc() -> None:
            dlg._emit("close", True)
            on_close()

        dlg._esc_handler = _esc
        return dlg

    @staticmethod
    def error(
        title: str,
        message: str,
        parent: urwid.Widget,
        actions: Sequence[tuple[str, Callable[[], None]]],
        esc_action_index: int = -1,
    ) -> "ModalDialog":
        dlg: ModalDialog | None = None

        def _make_handler(cb: Callable[[], None]):
            def _press(_btn: urwid.Button) -> None:
                assert dlg is not None
                dlg._emit("close", True)
                cb()

            return _press

        buttons = []
        for label, cb in actions:
            btn = urwid.AttrMap(
                urwid.Button(label, on_press=_make_handler(cb)),
                None,
                focus_map="btn_focus",
            )
            buttons.append(("weight", 1, btn))

        body = urwid.Pile(
            [
                urwid.Text(message),
                urwid.Divider(),
                urwid.Columns(buttons, dividechars=2),
            ]
        )
        dlg = ModalDialog(title, body, parent)

        esc_idx = esc_action_index if esc_action_index >= 0 else len(actions) - 1
        esc_cb = actions[esc_idx][1] if 0 <= esc_idx < len(actions) else None

        if esc_cb is not None:

            def _esc() -> None:
                dlg._emit("close", True)
                esc_cb()

            dlg._esc_handler = _esc
        return dlg

    @staticmethod
    def input_text(
        title: str,
        message: str,
        parent: urwid.Widget,
        on_save: "Callable[[str], None]",
        on_cancel: "Callable[[], None] | None" = None,
        default: str = "",
    ) -> "ModalDialog":
        dlg: ModalDialog | None = None
        edit = urwid.Edit(caption="  ", edit_text=default)

        def _save(btn: urwid.Button) -> None:
            assert dlg is not None
            dlg._emit("close", True)
            on_save(edit.get_edit_text())

        def _cancel(btn: urwid.Button) -> None:
            assert dlg is not None
            dlg._emit("close", False)
            if on_cancel:
                on_cancel()

        save_btn = urwid.AttrMap(
            urwid.Button("Save", on_press=_save), None, focus_map="btn_focus"
        )
        cancel_btn = urwid.AttrMap(
            urwid.Button("Cancel", on_press=_cancel), None, focus_map="btn_focus"
        )

        body = urwid.Pile(
            [
                urwid.Text(message),
                urwid.Divider(),
                urwid.AttrMap(edit, "input"),
                urwid.Divider(),
                urwid.Columns(
                    [("weight", 1, save_btn), ("weight", 1, cancel_btn)],
                    dividechars=2,
                ),
            ]
        )
        dlg = ModalDialog(title, body, parent)

        def _esc() -> None:
            dlg._emit("close", False)
            if on_cancel:
                on_cancel()

        dlg._esc_handler = _esc
        return dlg


class FormDialog(ModalDialog):
    """A modal dialog. 
    Esc triggers ``on_cancel``."""

    def __init__(
        self,
        title: str,
        form,
        parent: urwid.Widget,
        buttons: Sequence[tuple[str, Callable[[], None]]],
        *,
        section: str | None = None,
        on_cancel: Callable[[], None] | None = None,
        width: int = 50,
        min_width: int = 40,
    ) -> None:
        self._form = form
        self._error = urwid.Text("")

        def _press(cb: Callable[[], None]):
            return lambda _btn: cb()

        btn_row = urwid.Columns(
            [
                (
                    "weight",
                    1,
                    urwid.AttrMap(
                        urwid.Button(label, on_press=_press(cb)),
                        None,
                        focus_map="btn_focus",
                    ),
                )
                for label, cb in buttons
            ],
            dividechars=2,
        )

        items: list = []
        if section is not None:
            items.extend([
                ("pack", urwid.Text(("section", f" {section}"), align="left")),
                ("pack", urwid.Divider("-")),
            ])
        items.extend([
            ("pack", form),
            ("pack", urwid.Divider()),
            ("pack", self._error),
            ("pack", urwid.Divider()),
            ("pack", btn_row),
        ])

        super().__init__(
            title, urwid.Pile(items), parent, width=width, min_width=min_width
        )
        if on_cancel is not None:
            self._esc_handler = on_cancel

    def set_error(self, message: str) -> None:
        self._error.set_text(("form_error", message))

    def validated_values(self) -> dict | None:
        """Form values, or ``None`` after displaying validation errors."""
        errors = self._form.validate()
        if errors:
            self.set_error("; ".join(errors))
            return None
        return self._form.get_values()
