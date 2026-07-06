import urwid

from rapmat.tui.widgets.form import FormGroup, checkbox_field, text_field


def test(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fields = [
        checkbox_field("grid_search", "Compositional Grid Search", default=False),
        text_field("formula", "Formula"),
    ]
    form = FormGroup(fields)

    def _sync(widget=None, state=None):
        vals = form.get_values()
        is_grid = vals.get("grid_search", False)

        form.set_field_disabled("formula", is_grid)


        with open("ui_log.txt", "a") as f:
            f.write(
                f"Sync called. Grid from vals={is_grid}, state from callback={state}\n"
            )

    grid_cb = form.get_widget("grid_search")
    urwid.connect_signal(grid_cb, "change", _sync)

    grid_cb.keypress((10,), " ")
    grid_cb.keypress((10,), " ")
    grid_cb.keypress((10,), " ")
    grid_cb.keypress((10,), " ")
