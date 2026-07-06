"""Unit tests for the typed search/run config model."""

import json

from rapmat.core.config import SearchConfig, merge_config_dicts


def test_defaults_are_study_facing():
    """Canonical defaults match the study-create form, not the old csp.py fallbacks."""
    cfg = SearchConfig()
    assert cfg.symprec == 1e-2
    assert cfg.force_conv_crit == 5e-3
    assert cfg.steps_max == 2000
    assert cfg.domain == "bulk"
    assert cfg.calculator == "MATTERSIM"
    assert cfg.min_dist == 0.5
    assert cfg.forces_break == 1000.0
    assert cfg.max_count == 10
    assert cfg.pressure_gpa == 0.0
    assert cfg.formula == {}
    assert cfg.seed is None


def test_merge_precedence():
    """Batch overrides study, study columns injected only when truthy."""
    study_cfg = {"symprec": 1e-3, "pressure_gpa": 5.0}
    batch_cfg = {"symprec": 1e-4, "formula": {"Al": 2, "O": 3}}

    merged = merge_config_dicts(
        study_cfg, batch_cfg, system="Al-O", domain="bulk", calculator="MATTERSIM"
    )

    assert merged["symprec"] == 1e-4

    assert merged["pressure_gpa"] == 5.0

    assert merged["system"] == "Al-O"
    assert merged["domain"] == "bulk"
    assert merged["calculator"] == "MATTERSIM"

    assert "steps_max" not in merged


    bare = merge_config_dicts(study_cfg, batch_cfg)
    assert "system" not in bare
    assert "domain" not in bare
    assert "calculator" not in bare


def test_from_stored_is_typed_merge():
    cfg = SearchConfig.from_stored(
        {"symprec": 1e-3}, {"formula": {"Cu": 1}}, system="Cu", domain="bulk"
    )
    assert isinstance(cfg, SearchConfig)
    assert cfg.symprec == 1e-3
    assert cfg.formula == {"Cu": 1}
    assert cfg.system == "Cu"



def test_roundtrip_old_style_dict_preserves_numbers():
    """Stored numbers survive a json.loads -> model_validate round-trip verbatim."""
    old_config_json = json.dumps(
        {
            "formula": {"Al": 2, "O": 3},
            "symprec": 1.234e-4,
            "force_conv_crit": 4.2e-2,
            "steps_max": 777,
            "pressure_gpa": 3.5,
            "seed": 42,
        }
    )
    cfg = SearchConfig.model_validate(json.loads(old_config_json))
    assert cfg.symprec == 1.234e-4
    assert cfg.force_conv_crit == 4.2e-2
    assert cfg.steps_max == 777
    assert cfg.pressure_gpa == 3.5
    assert cfg.seed == 42
    assert cfg.formula == {"Al": 2, "O": 3}


def test_legacy_keys_ignored():
    """Retired keys validate without error and don't appear in the model."""
    cfg = SearchConfig.model_validate(
        {"dedup": False, "dedup_threshold": 5.0, "symprec": 1e-3}
    )
    assert cfg.symprec == 1e-3
    dumped = cfg.model_dump()
    assert "dedup" not in dumped
    assert "dedup_threshold" not in dumped


def test_missing_keys_get_canonical_defaults():
    assert SearchConfig.model_validate({}).symprec == 1e-2

    assert SearchConfig.model_validate({"symprec": 1e-4}).symprec == 1e-4


def test_tuple_formula_units_coerced_to_list():
    cfg = SearchConfig.model_validate({"formula_units": (2, 4)})
    assert cfg.formula_units == [2, 4]
