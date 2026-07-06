import urwid


class SubmitSearchEdit(urwid.Edit):
    """Search field that submits its text on enter and exits on esc."""

    def __init__(self, on_search, on_exit) -> None:
        super().__init__(caption="Search: ")
        self._on_search = on_search
        self._on_exit = on_exit

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "enter":
            self._on_search(self.edit_text)
            self._on_exit()
            return None
        if key == "esc":
            self._on_exit()
            return None
        return super().keypress(size, key)


class LiveSearchEdit(urwid.Edit):
    """Search field that re-filters on every keystroke.

    Esc calls ``on_exit`` (caller cancels the filter), enter calls
    ``on_submit`` if given (caller keeps the filter).
    """

    def __init__(self, on_change, on_exit, on_submit=None) -> None:
        super().__init__(caption="Search: ")
        self._on_change = on_change
        self._on_exit = on_exit
        self._on_submit = on_submit

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "esc":
            self._on_exit()
            return None
        if key == "enter" and self._on_submit is not None:
            self._on_submit()
            return None
        result = super().keypress(size, key)
        self._on_change(self.get_edit_text())
        return result
