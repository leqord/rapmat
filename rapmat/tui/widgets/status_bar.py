import urwid

from rapmat.tui.keymap import FooterHint


class StatusBar(urwid.WidgetWrap):
    def __init__(self) -> None:
        self._hints: list[FooterHint] = []
        self._message: str = ""
        self._text = urwid.Text("", wrap="clip")
        self._last_maxcol: int | None = None
        super().__init__(urwid.AttrMap(self._text, "footer"))

    def set_hints(self, hints: list[FooterHint]) -> None:
        self._hints = list(hints)
        self._refit()

    def set_message(self, msg: str) -> None:
        self._message = msg
        self._refit()

    def clear_message(self) -> None:
        self.set_message("")

    # ------------------------------------------------------------------ #
    #  Fitting
    # ------------------------------------------------------------------ #

    def render(self, size: tuple, focus: bool = False):
        (maxcol,) = size
        if maxcol != self._last_maxcol:
            self._last_maxcol = maxcol
            self._refit()
        return super().render(size, focus)

    def _refit(self) -> None:
        maxcol = self._last_maxcol
        if maxcol is None:
            self._text.set_text(self._compose_hints(self._hints))
            return

        msg = f" {self._message} " if self._message else ""
        budget = maxcol - len(msg)

        by_priority = sorted(
            range(len(self._hints)), key=lambda i: self._hints[i][2]
        )
        kept: set[int] = set()
        used = 0
        for i in by_priority:
            cost = len(self._part(self._hints[i])) + 2
            if used + cost <= budget:
                kept.add(i)
                used += cost

        line = self._compose_hints(
            [h for i, h in enumerate(self._hints) if i in kept]
        )
        if msg:
            pad = max(1, maxcol - len(line) - len(msg))
            line = line + " " * pad + msg
        self._text.set_text(line)

    @staticmethod
    def _part(hint: FooterHint) -> str:
        key, label, _priority = hint
        return f"[{key}] {label}"

    def _compose_hints(self, hints: list[FooterHint]) -> str:
        return "  " + "  ".join(self._part(h) for h in hints) if hints else ""
