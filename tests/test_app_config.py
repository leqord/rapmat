"""Tests for local application settings."""

import pytest

from rapmat import app_config


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(app_config, "APP_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(app_config, "_SETTINGS_FILE", tmp_path / "settings.toml")
    for name in app_config._VASP_COMMAND_ENV:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


class TestResolveVaspCommand:
    def test_empty_by_default(self):
        assert app_config.resolve_vasp_command() == ""

    def test_reads_ase_env_var(self, monkeypatch):
        monkeypatch.setenv("ASE_VASP_COMMAND", "srun vasp_std")
        assert app_config.resolve_vasp_command() == "srun vasp_std"

    def test_reads_vasp_command_env_var(self, monkeypatch):
        monkeypatch.setenv("VASP_COMMAND", "vasp_gam")
        assert app_config.resolve_vasp_command() == "vasp_gam"

    def test_saved_value_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("ASE_VASP_COMMAND", "from-env")
        app_config.persist_vasp_command("from-config")
        assert app_config.resolve_vasp_command() == "from-config"

    def test_ignores_vasp_script(self, monkeypatch):
        monkeypatch.setenv("VASP_SCRIPT", "/opt/run_vasp.py")
        assert app_config.resolve_vasp_command() == ""


class TestPersistVaspCommand:
    def test_round_trips(self):
        assert app_config.persist_vasp_command("mpirun -np 8 vasp_std") is True
        assert app_config.resolve_vasp_command() == "mpirun -np 8 vasp_std"

    def test_reports_no_change_when_identical(self):
        app_config.persist_vasp_command("vasp_std")
        assert app_config.persist_vasp_command("vasp_std") is False

    def test_blank_is_not_persisted(self):
        assert app_config.persist_vasp_command("   ") is False
        assert not (app_config.settings_file_path()).exists()

    def test_does_not_clobber_other_sections(self, isolated_settings):
        (isolated_settings / "settings.toml").write_text(
            '[other]\nkeep = "me"\n', encoding="utf-8"
        )
        app_config.persist_vasp_command("vasp_std")
        assert app_config.load_app_settings()["other"]["keep"] == "me"


class TestIsolation:
    def test_the_real_config_dir_is_never_used(self):
        """Regression"""
        from rapmat import config as rapmat_config

        assert app_config.APP_CONFIG_DIR != rapmat_config.APP_CONFIG_DIR
        assert (
            rapmat_config.APP_CONFIG_DIR
            not in app_config.settings_file_path().parents
        )


class TestLoadAppSettings:
    def test_missing_file_is_empty(self):
        assert app_config.load_app_settings() == {}

    def test_malformed_file_does_not_raise(self, isolated_settings):
        (isolated_settings / "settings.toml").write_text(
            "this is not [valid toml", encoding="utf-8"
        )
        assert app_config.load_app_settings() == {}
        assert app_config.resolve_vasp_command() == ""
