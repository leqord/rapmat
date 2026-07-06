import urwid

from rapmat.core.config import SearchConfig
from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.widgets.table import SortableTable
from rapmat.utils import hardware


def _format_run_status(counts: dict[str, int]) -> str:
    parts: list[str] = []
    relaxed = counts.get("relaxed", 0)
    generating = counts.get("generating", 0)
    generated = counts.get("generated", 0)
    discarded = counts.get("discarded", 0)
    error = counts.get("error", 0)
    if relaxed:
        parts.append(f"{relaxed} relaxed")
    if generating:
        parts.append(f"{generating} generating")
    if generated:
        parts.append(f"{generated} pending")
    if discarded:
        parts.append(f"{discarded} discarded")
    if error:
        parts.append(f"{error} error")
    return " . ".join(parts) if parts else "-"


def _format_run_row(run: dict) -> list[str]:
    formula_map = SearchConfig.model_validate(run.get("config", {})).formula
    formula = (
        "".join(f"{el}{n}" if n > 1 else el for el, n in formula_map.items())
        if formula_map
        else "-"
    )
    ts = run.get("timestamp", "")[:16].replace("T", " ")
    status = run.get("_status_summary", "-")
    return [run.get("name", "-"), str(run.get("domain", "-")), formula, ts, status]


_RECENT_COLS = [
    ("Name", 20),
    ("Domain", 10),
    ("Formula", 12),
    ("Created", 16),
    ("Structures", 28),
]


class HomeScreen(ScreenBase):
    title = "Home"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._widget: urwid.Widget | None = None
        self._db_info_text: urwid.Text | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._state.refresh_runs_if_needed()
        self._widget = self._build_widget()
        self.refresh_footer()
        return self._widget

    def on_resume(self) -> None:
        self._state.refresh_runs_if_needed()
        super().on_resume()
        if self._widget is not None:
            self._refresh_recent_table()

    def bindings(self) -> list[KeyBinding]:
        return [
            KeyBinding(
                ("n",), "New Run", self._go_new_run,
                help="Start a new CSP run", priority=10,
            ),
            KeyBinding(
                ("s",), "Studies", self._go_studies,
                help="Browse studies", priority=20,
            ),
            KeyBinding(
                ("p",), "Phonon", self._go_phonon,
                help="Phonon dispersion tool", priority=30,
            ),
            KeyBinding(
                ("d",), "DB", self._go_db_settings,
                help="Database settings", priority=40,
            ),
            KeyBinding(
                ("i",), "Status", self._go_status,
                help="Locks and app status", priority=50,
            ),
            KeyBinding(
                ("q",), "Quit", self._do_quit,
                help="Quit rapmat", priority=60,
            ),
        ]

    def esc_label(self) -> str:
        return ""

    # ------------------------------------------------------------------ #
    #  Layout helpers
    # ------------------------------------------------------------------ #

    def _build_widget(self) -> urwid.Widget:
        def _btn(label: str, callback) -> urwid.Widget:
            btn = urwid.Button(label, on_press=callback)
            return urwid.AttrMap(btn, "menu_item", focus_map="menu_focus")

        def _section(label: str) -> list[urwid.Widget]:
            return [urwid.Divider(), urwid.Text(("section", f" {label}"), align="left")]

        actions = urwid.Pile(
            [
                urwid.Text(("section", " Quick Actions"), align="left"),
                *_section("CSP"),
                _btn("[N] New CSP Run", self._go_new_run),
                *_section("Studies"),
                _btn("[S] Studies", self._go_studies),
                *_section("Tools"),
                _btn("[P] Phonon", self._go_phonon),
                *_section("Settings"),
                _btn("[D] DB Settings", self._go_db_settings),
                _btn("[I] Status", self._go_status),
                urwid.Divider("-"),
                _btn("[Q] Quit", self._do_quit),
            ]
        )

        left_panel = urwid.ListBox(
            urwid.SimpleListWalker(
                [
                    actions,
                    urwid.Divider(),
                    self._build_db_info(),
                ]
            )
        )

        recent_data = self._get_recent_runs()
        self._recent_table = SortableTable(
            columns=_RECENT_COLS,
            row_data=recent_data,
            format_row=_format_run_row,
            on_focus_change=None,
        )
        urwid.connect_signal(self._recent_table, "select", self._on_run_select)

        right_panel = urwid.Pile(
            [
                (
                    "pack",
                    urwid.Text(
                        ("section", " Recent Runs")
                    ),
                ),
                ("pack", urwid.Divider("-")),
                ("weight", 1, self._recent_table),
            ]
        )

        columns = urwid.Columns(
            [
                ("weight", 1, urwid.Padding(left_panel, left=2, right=2)),
                ("weight", 3, urwid.Padding(right_panel, left=1, right=1)),
            ],
            dividechars=1,
        )

        return columns

    def _build_db_info(self) -> urwid.Widget:
        self._db_info_text = urwid.Text(self._db_info_lines())
        return self._db_info_text

    def _db_info_lines(self) -> list[tuple[str, str]]:
        url = self._state.store.get_url() or "-"
        hw = hardware.home_label(hardware.cached())
        return [
            ("section", " Database\n"),
            ("details", f" {url}\n\n"),
            ("section", " Hardware\n"),
            ("details", f" {hw}"),
        ]

    def refresh_hw_label(self) -> None:
        """Re-render the DB-info panel once background detection completes."""
        if self._db_info_text is not None:
            self._db_info_text.set_text(self._db_info_lines())

    def _get_recent_runs(self) -> list[dict]:
        runs = sorted(
            self._state.runs_cache,
            key=lambda r: r.timestamp,
            reverse=True,
        )[:5]
        enriched = []
        for run in runs:
            counts = self._state.store.count_by_status(run.name)
            r = run.model_dump()
            r["_status_summary"] = _format_run_status(counts)
            enriched.append(r)
        return enriched

    def _refresh_recent_table(self) -> None:
        recent_data = self._get_recent_runs()
        self._recent_table.set_data(recent_data)

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def _go_new_run(self, _btn: urwid.Button | None = None) -> None:
        from rapmat.tui.screens.csp_search import CSPSearchScreen

        self._router.push(CSPSearchScreen(self._state, self._router))

    def _go_studies(self, _btn: urwid.Button | None = None) -> None:
        from rapmat.tui.screens.study_list import StudyListScreen

        self._router.push(StudyListScreen(self._state, self._router))

    def _go_phonon(self, _btn: urwid.Button | None = None) -> None:
        from rapmat.tui.screens.phonon import PhononDispersionScreen

        self._router.push(PhononDispersionScreen(self._state, self._router))

    def _go_db_settings(self, _btn: urwid.Button | None = None) -> None:
        from rapmat.tui.screens.db_settings import DbSettingsScreen

        self._router.push(DbSettingsScreen(self._state, self._router))

    def _go_status(self, _btn: urwid.Button | None = None) -> None:
        from rapmat.tui.screens.status import StatusScreen

        self._router.push(StatusScreen(self._state, self._router))

    def _do_quit(self, _btn: urwid.Button | None = None) -> None:
        if self._state.request_quit is not None:
            self._state.request_quit()
        else:
            raise urwid.ExitMainLoop()

    def _on_run_select(self, _table, run: dict) -> None:
        self._state.active_run = run["name"]
        from rapmat.tui.screens.results import ResultsScreen

        self._router.push(ResultsScreen(self._state, self._router))

