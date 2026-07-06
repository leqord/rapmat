"""Calculator availability probing, including broken installs."""

import importlib.util

import rapmat.calculators as calcs
from rapmat.tui.screens.status import _format_calc_row, _load_calc_rows

_WARP_ERROR = "module 'warp._src.utils' has no attribute 'x'"


def test_probe_missing_module_has_no_error(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    available, error = calcs.probe_calculator(calcs.Calculators.UPET)
    assert available is False
    assert error is None


def test_probe_broken_import_reports_error(monkeypatch):
    def _boom(name):
        raise AttributeError(_WARP_ERROR)

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    available, error = calcs.probe_calculator(calcs.Calculators.UPET)
    assert available is False
    assert error == f"AttributeError: {_WARP_ERROR}"
    assert calcs.is_calculator_available(calcs.Calculators.UPET) is False


def test_status_rows_distinguish_broken_from_missing(monkeypatch):
    def _fake_probe(calc):
        if calc is calcs.Calculators.UPET:
            return False, f"AttributeError: {_WARP_ERROR}"
        if calc is calcs.Calculators.VASP:
            return False, None
        return True, None

    monkeypatch.setattr(calcs, "probe_calculator", _fake_probe)
    rows = {r["name"]: r for r in _load_calc_rows()}

    assert _format_calc_row(rows["UPET"])[1] == "broken"
    assert _format_calc_row(rows["VASP"])[1] == "not found"
    assert _format_calc_row(rows["MATTERSIM"])[1] == "installed"
    assert _WARP_ERROR in rows["UPET"]["error"]
