import urwid

from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.widgets.config_grid import build_config_grid
from rapmat.tui.widgets.search import LiveSearchEdit
from rapmat.tui.widgets.table import SortableTable

_STUDY_COLS = [
    ("Name", 22),
    ("System", 14),
    ("Domain", 12),
    ("Calculator", 16),
    ("Created", 16),
    ("Runs", 6),
]


def _enrich_studies(state: "AppState") -> list[dict]:
    enriched = []
    for s in state.studies_cache:
        runs = state.store.get_study_runs(s.study_id)
        d = s.to_dict()
        d["_run_count"] = len(runs)
        enriched.append(d)
    return sorted(enriched, key=lambda s: s.get("timestamp", ""), reverse=True)


def _format_study_row(row: dict) -> list[str]:
    ts = row.get("timestamp", "")[:16].replace("T", " ")
    return [
        row.get("study_id", "-"),
        row.get("system", "-"),
        row.get("domain", "-"),
        row.get("calculator", "-"),
        ts,
        str(row.get("_run_count", 0)),
    ]


class StudyListScreen(ScreenBase):
    title = "Studies"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._all_rows: list[dict] = []
        self._table: SortableTable | None = None
        self._search_edit: LiveSearchEdit | None = None
        self._footer_pile: urwid.Pile | None = None
        self._sort_col: int = 0
        self._searching: bool = False
        self._widget: urwid.Widget | None = None
        self._body_pile: urwid.Pile | None = None
        self._details_content: urwid.WidgetPlaceholder | None = None
        self._details_panel: urwid.Widget | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._widget = urwid.WidgetPlaceholder(urwid.SolidFill())
        self._state.refresh_studies_if_needed()
        self._widget.original_widget = self._build_widget()
        self.refresh_footer()
        return self._widget

    def on_resume(self) -> None:
        self._state.refresh_studies_if_needed()
        super().on_resume()
        if self._table is not None:
            self._all_rows = _enrich_studies(self._state)
            self._table.set_data(self._all_rows)

    def bindings(self) -> list[KeyBinding]:
        return [
            KeyBinding(
                ("/",), "Search", self._enter_search,
                help="Filter studies", priority=10,
            ),
            KeyBinding(
                ("n",), "New", self._open_create,
                help="Create a new study", priority=20,
            ),
            KeyBinding(
                ("s",), "Sort", self._cycle_sort,
                help="Change the sort column", priority=30,
            ),
            KeyBinding(
                ("delete",), "Remove", self._delete_focused,
                help="Delete the focused study",
                priority=40,
            ),
        ]

    def extra_hints(self) -> list:
        return [("Enter", "Open", 5)]

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _build_widget(self) -> urwid.Widget:
        self._all_rows = _enrich_studies(self._state)

        self._table = SortableTable(
            columns=_STUDY_COLS,
            row_data=self._all_rows,
            format_row=_format_study_row,
            on_focus_change=self._on_study_focus_change,
        )
        urwid.connect_signal(self._table, "select", self._on_study_select)

        self._details_content = urwid.WidgetPlaceholder(
            urwid.Text("No study selected.")
        )
        self._details_panel = urwid.LineBox(
            self._details_content,
            title="Study Configuration",
        )

        self._search_edit = LiveSearchEdit(
            on_change=self._apply_search,
            on_exit=self._exit_search,
            on_submit=self._submit_search,
        )

        self._footer_pile = urwid.Pile([])

        self._body_pile = urwid.Pile(
            [
                ("weight", 1, self._table),
                ("pack", urwid.Divider()),
                ("pack", self._details_panel),
                ("pack", urwid.Divider()),
                ("pack", self._footer_pile),
            ]
        )
        if self._table:
            self._on_study_focus_change(self._table.get_focused_row())
        return urwid.Padding(self._body_pile, left=1, right=1)

    # ------------------------------------------------------------------ #
    #  Search helpers
    # ------------------------------------------------------------------ #

    def _apply_search(self, query: str) -> None:
        if not query:
            self._table.set_data(self._all_rows)
            return
        q = query.lower()
        filtered = [
            r
            for r in self._all_rows
            if q in r.get("study_id", "").lower()
            or q in r.get("system", "").lower()
            or q in r.get("domain", "").lower()
            or q in r.get("calculator", "").lower()
        ]
        self._table.set_data(filtered)

    def _enter_search(self) -> None:
        if (
            self._search_edit is None
            or self._footer_pile is None
            or self._body_pile is None
        ):
            return
        self._searching = True
        self._search_edit.set_edit_text("")
        self._footer_pile.contents = [
            (self._search_edit, self._footer_pile.options()),
        ]
        self._footer_pile.focus_position = 0

        self._body_pile.focus_position = len(self._body_pile.contents) - 1

    def _leave_search_mode(self) -> None:
        self._searching = False
        if self._body_pile is not None:
            self._body_pile.focus_position = 0
        if self._footer_pile is not None:
            self._footer_pile.contents = []

    def _exit_search(self) -> None:
        self._leave_search_mode()
        self._apply_search("")

    def _submit_search(self) -> None:
        self._leave_search_mode()

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def _on_study_focus_change(self, study: dict | None) -> None:
        if getattr(self, "_details_content", None) is None:
            return

        if study is None:
            self._details_content.original_widget = urwid.Text(
                [("details", "No study selected.")]
            )
        else:
            config = study.get("config", {})
            self._details_content.original_widget = build_config_grid(config)

    def _on_study_select(self, _table, study: dict) -> None:
        self._state.active_study = study["study_id"]
        from rapmat.tui.screens.study_detail import StudyDetailScreen

        self._router.push(StudyDetailScreen(self._state, self._router))

    def _open_create(self) -> None:
        from rapmat.tui.screens.study_create import StudyCreateScreen

        self._router.push(StudyCreateScreen(self._state, self._router))

    def _cycle_sort(self) -> None:
        self._sort_col = (self._sort_col + 1) % len(_STUDY_COLS)
        if self._table:
            self._table.sort_by(self._sort_col)

    def _delete_focused(self) -> None:
        if self._table is not None:
            study = self._table.get_focused_row()
            if study:
                self._open_delete_modal(study["study_id"])

    def _dialog_host_get(self) -> "urwid.Widget | None":
        return self._widget.original_widget if self._widget is not None else None

    def _dialog_host_set(self, widget: urwid.Widget) -> None:
        self._widget.original_widget = widget

    def _open_delete_modal(self, study_id: str) -> None:
        def _confirmed() -> None:
            self._state.store.delete_study(study_id)
            self._state.invalidate_studies()
            self._state.refresh_studies_if_needed()
            self._all_rows = _enrich_studies(self._state)
            if self._table is not None:
                self._table.set_data(self._all_rows)
                self._on_study_focus_change(self._table.get_focused_row())

        self.confirm_dialog(
            "Delete Study",
            (
                f"Are you sure you want to permanently delete study '{study_id}'?\n\n"
            ),
            _confirmed,
        )
