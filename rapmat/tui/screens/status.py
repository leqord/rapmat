import urwid

from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.widgets.table import SortableTable

_CALC_COLS = [
    ("Calculator", 20),
    ("Status", 14),
    ("Description", 50),
]


def _load_calc_rows() -> list[dict]:
    import rapmat.calculators as calcs

    rows = []
    for calc in calcs.Calculators:
        available, error = calcs.probe_calculator(calc)
        rows.append(
            {
                "name": calc.value,
                "available": available,
                "error": error,
                "description": calcs.CALCULATOR_META[calc].get("description", ""),
            }
        )
    return rows


def _format_calc_row(row: dict) -> list[str]:
    if row["available"]:
        status = "installed"
    else:
        status = "broken" if row["error"] else "not found"
    return [row["name"], status, row["description"]]


def _calc_attr(row: dict) -> str:
    return "success" if row["available"] else "error"


class StatusScreen(ScreenBase):
    title = "Status"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._table: SortableTable | None = None
        self._errors_text: urwid.Text | None = None
        self._widget: urwid.Widget | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._widget = self._build_widget()
        self.refresh_footer()
        return self._widget

    def bindings(self) -> list[KeyBinding]:
        return [
            KeyBinding(
                ("r",), "Refresh", self._action_refresh,
                help="Re-probe the installed calculators", priority=10,
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _build_widget(self) -> urwid.Widget:
        import platformdirs

        _APP_NAME = "rapmat-materials"
        APP_CONFIG_DIR = platformdirs.user_config_dir(_APP_NAME)
        APP_DATA_DIR = platformdirs.user_data_dir(_APP_NAME)

        rows = _load_calc_rows()
        self._table = SortableTable(
            columns=_CALC_COLS,
            row_data=rows,
            format_row=_format_calc_row,
            attr_fn=_calc_attr,
        )
        self._errors_text = urwid.Text("")
        self._update_errors(rows)

        paths_text = urwid.Text(
            [
                ("section", " Application Paths\n"),
                ("form_label", "  Config: "),
                ("details", str(APP_CONFIG_DIR) + "\n"),
                ("form_label", "  Data:   "),
                ("details", str(APP_DATA_DIR)),
            ]
        )

        body = urwid.Pile(
            [
                ("pack", urwid.Text(("section", " Calculators"), align="left")),
                ("pack", urwid.Divider("-")),
                ("weight", 1, self._table),
                ("pack", self._errors_text),
                ("pack", urwid.Divider()),
                ("pack", paths_text),
                ("pack", urwid.Divider())
            ]
        )

        return urwid.Padding(body, left=1, right=1)

    def _update_errors(self, rows: list[dict]) -> None:
        if self._errors_text is None:
            return
        markup: list = []
        for row in rows:
            if row["error"]:
                markup.extend([
                    ("form_label", f"  {row['name']}: "),
                    ("error", row["error"] + "\n"),
                ])
        if markup:
            self._errors_text.set_text(
                [("section", " Import Errors\n")] + markup
            )
        else:
            self._errors_text.set_text("")

    # ------------------------------------------------------------------ #
    #  Input
    # ------------------------------------------------------------------ #

    def _action_refresh(self) -> None:
        if self._table is not None:
            rows = _load_calc_rows()
            self._table.set_data(rows)
            self._update_errors(rows)
