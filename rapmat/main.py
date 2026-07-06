def app_main() -> None:
    from rapmat.utils.console import configure_logging

    configure_logging()

    from rapmat.db_config import resolve_store
    from rapmat.tui.app import RapmatApp
    from rapmat.tui.state import AppState

    startup_error: Exception | None = None
    try:
        store = resolve_store()
    except Exception as exc:
        startup_error = exc
        from rapmat.storage.sqlite_store import SQLiteStore

        store = SQLiteStore(":memory:")

    state = AppState(store=store)
    RapmatApp(state, startup_error=startup_error).run()
