from pathlib import Path

from rapmat import app_config
from rapmat.utils.common import workdir_context


class TestTemporaryByDefault:
    def test_yields_a_directory(self):
        with workdir_context(None) as wdir:
            assert wdir.is_dir()

    def test_is_removed_on_exit(self):
        with workdir_context(None) as wdir:
            kept = wdir
        assert not kept.exists()


class TestConfiguredRoot:
    def test_sits_under_the_configured_root(self, tmp_path):
        root = tmp_path / "calcs"
        app_config.persist_calc_root(str(root))

        with workdir_context(None) as wdir:
            assert root in wdir.parents

    def test_survives_the_context(self, tmp_path):
        app_config.persist_calc_root(str(tmp_path / "calcs"))

        with workdir_context(None) as wdir:
            kept = wdir
        assert kept.is_dir()

    def test_missing_root_is_created(self, tmp_path):
        root = tmp_path / "a" / "b" / "calcs"
        app_config.persist_calc_root(str(root))

        with workdir_context(None):
            assert root.is_dir()

    def test_sessions_do_not_collide(self, tmp_path):
        app_config.persist_calc_root(str(tmp_path / "calcs"))

        with workdir_context(None) as first, workdir_context(None) as second:
            assert first != second

    def test_hint_appears_in_the_session_name(self, tmp_path):
        app_config.persist_calc_root(str(tmp_path / "calcs"))

        with workdir_context(None, session_hint="worker-7") as wdir:
            assert "worker-7" in wdir.name

    def test_clearing_the_root_restores_temp_dirs(self, tmp_path):
        app_config.persist_calc_root(str(tmp_path / "calcs"))
        app_config.persist_calc_root("")

        with workdir_context(None) as wdir:
            kept = wdir
        assert not kept.exists()


class TestExplicitWorkdir:
    def test_wins_over_the_configured_root(self, tmp_path):
        app_config.persist_calc_root(str(tmp_path / "calcs"))
        target = tmp_path / "explicit"

        with workdir_context(str(target)) as wdir:
            assert wdir == target.resolve()

    def test_is_created_and_kept(self, tmp_path):
        target = tmp_path / "explicit"

        with workdir_context(str(target)) as wdir:
            assert wdir.is_dir()
        assert Path(target).is_dir()
