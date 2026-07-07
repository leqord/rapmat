


def test_tui():
    from rapmat.storage import SQLiteStore
    from rapmat.tui.app import RapmatApp
    from rapmat.tui.state import AppState

    store = SQLiteStore(":memory:")
    state = AppState(store=store)
    app = RapmatApp(state)

    from rapmat.tui.screens.csp_search import CSPSearchScreen
    from rapmat.tui.screens.study_create import StudyCreateScreen

    print("Building StudyCreateScreen...")
    s1 = StudyCreateScreen(state, app._router)
    s1.build()
    print("StudyCreateScreen built successfully.")

    print("Building CSPSearchScreen...")
    s2 = CSPSearchScreen(state, app._router)
    s2.build()
    print("CSPSearchScreen built successfully.")


if __name__ == "__main__":
    test_tui()
