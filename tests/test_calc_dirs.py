from pathlib import Path, PurePosixPath
from unittest.mock import patch

from ase.build import bulk

from rapmat.calculators.calc_dirs import (CalcDirAllocator,
                                          prune_calculation_dir,
                                          sanitize_label)


def _write(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("x", encoding="utf-8")


class TestSanitizeLabel:
    def test_structure_id_becomes_nested_parts(self):
        assert sanitize_label("run-a/12") == PurePosixPath("run-a/12")

    def test_backslashes_split_too(self):
        assert sanitize_label(r"run-a\12") == PurePosixPath("run-a/12")

    def test_parent_refs_are_dropped(self):
        assert sanitize_label("../../etc/passwd") == PurePosixPath("etc/passwd")

    def test_absolute_posix_path_cannot_escape(self):
        result = sanitize_label("/etc/x")
        assert not str(result).startswith("/")
        assert result == PurePosixPath("etc/x")

    def test_drive_letter_cannot_escape(self):
        result = sanitize_label("C:/windows/x")
        assert ":" not in str(result)
        assert result == PurePosixPath("C_/windows/x")

    def test_illegal_characters_are_replaced(self):
        assert sanitize_label("a b*c?") == PurePosixPath("a_b_c_")

    def test_empty_label_still_yields_a_component(self):
        assert sanitize_label("") == PurePosixPath("_")

    def test_long_parts_are_capped(self):
        assert len(sanitize_label("x" * 500).parts[0]) == 120


class TestSequence:
    def test_calls_are_siblings(self, tmp_path):
        alloc = CalcDirAllocator(tmp_path)
        alloc.set_label("r/1")
        first, second = alloc.next(), alloc.next()

        assert first == tmp_path / "r" / "1" / "0001"
        assert second == tmp_path / "r" / "1" / "0002"

    def test_new_label_restarts_numbering(self, tmp_path):
        alloc = CalcDirAllocator(tmp_path)
        alloc.set_label("r/1")
        alloc.next()
        alloc.set_label("r/2")

        assert alloc.next() == tmp_path / "r" / "2" / "0001"

    def test_repeating_a_label_does_not_restart(self, tmp_path):
        alloc = CalcDirAllocator(tmp_path)
        alloc.set_label("r/1")
        first = alloc.next()
        alloc.set_label("r/1")

        assert alloc.next() != first

    def test_unlabelled_calls_sit_at_the_root(self, tmp_path):
        assert CalcDirAllocator(tmp_path).next() == tmp_path / "0001"

    def test_no_root_yields_nothing(self):
        alloc = CalcDirAllocator(None)
        alloc.set_label("r/1")
        assert alloc.next() is None

    def test_nothing_is_created_on_disk(self, tmp_path):
        root = tmp_path / "calc"
        alloc = CalcDirAllocator(root)
        alloc.set_label("r/1")
        alloc.next()
        assert not root.exists()


class TestPrune:
    def test_heavy_files_go_and_results_stay(self, tmp_path):
        _write(tmp_path, "WAVECAR", "CHGCAR", "OUTCAR", "INCAR")
        prune_calculation_dir(tmp_path)

        assert not (tmp_path / "WAVECAR").exists()
        assert not (tmp_path / "CHGCAR").exists()
        assert (tmp_path / "OUTCAR").exists()
        assert (tmp_path / "INCAR").exists()

    def test_missing_directory_is_a_no_op(self, tmp_path):
        assert prune_calculation_dir(tmp_path / "nope") == 0

    def test_none_is_a_no_op(self):
        assert prune_calculation_dir(None) == 0

    def test_allocating_prunes_the_previous_directory(self, tmp_path):
        alloc = CalcDirAllocator(tmp_path)
        alloc.set_label("r/1")

        first = alloc.next()
        _write(first, "WAVECAR", "OUTCAR")
        second = alloc.next()
        _write(second, "WAVECAR")

        assert not (first / "WAVECAR").exists()
        assert (first / "OUTCAR").exists()
        assert (second / "WAVECAR").exists(), "the live calculation must survive"

    def test_finalize_prunes_the_tail(self, tmp_path):
        alloc = CalcDirAllocator(tmp_path)
        last = alloc.next()
        _write(last, "WAVECAR", "OUTCAR")

        alloc.finalize()

        assert not (last / "WAVECAR").exists()
        assert (last / "OUTCAR").exists()

    def test_finalize_is_idempotent(self, tmp_path):
        alloc = CalcDirAllocator(tmp_path)
        _write(alloc.next(), "WAVECAR")
        alloc.finalize()
        alloc.finalize()


def _fake_vasp_class():
    from rapmat.calculators.vasp import RapmatVasp

    class FakeVasp(RapmatVasp):
        def initialize(self, atoms):
            ...

        def calculate(self, atoms=None, properties=("energy",), system_changes=None):
            target = Path(self.directory)
            target.mkdir(parents=True, exist_ok=True)
            for name in ("INCAR", "POSCAR", "OUTCAR", "WAVECAR", "CHGCAR"):
                (target / name).write_text("x", encoding="utf-8")
            self.atoms = atoms.copy()
            self.results = {"energy": -4.2, "free_energy": -4.2}

    return FakeVasp


class TestEvalLoopLayout:
    def _store(self, tmp_path):
        from conftest import add_relaxed_structure

        from rapmat.storage import SQLiteStore

        store = SQLiteStore.from_path(tmp_path / "layout_db")
        store.create_study(
            "s", system="Cu", domain="bulk", calculator="VASP", config={}
        )
        store.create_run(name="r", study_id="s")
        for i, a in enumerate((3.6, 3.7), start=1):
            add_relaxed_structure(
                store, "r", bulk("Cu", "fcc", a=a), -3.5, f"r/{i}"
            )
        return store

    def _run(self, tmp_path):
        from rapmat.calculators import Calculators
        from rapmat.calculators.factory import CalculatorProvider
        from rapmat.core.evaluation import run_eval_loop
        from rapmat.storage.status import StructureStatus

        store = self._store(tmp_path)
        pending = store.get_structures("r", status=StructureStatus.RELAXED)
        root = tmp_path / "calc"

        with patch("rapmat.calculators.vasp.RapmatVasp", _fake_vasp_class()):
            provider = CalculatorProvider(Calculators.VASP, root)
            run_eval_loop(pending, store, "r", provider, "VASP", "{}")

        return store, root

    def test_each_structure_gets_its_own_tree(self, tmp_path):
        _store, root = self._run(tmp_path)

        assert (root / "r" / "1" / "0001" / "OUTCAR").is_file()
        assert (root / "r" / "2" / "0001" / "OUTCAR").is_file()

    def test_inputs_are_archived_alongside_the_results(self, tmp_path):
        _store, root = self._run(tmp_path)

        leaf = root / "r" / "1" / "0001"
        assert (leaf / "INCAR").is_file()
        assert (leaf / "POSCAR").is_file()

    def test_heavy_binaries_are_pruned_everywhere(self, tmp_path):
        _store, root = self._run(tmp_path)

        assert list(root.rglob("WAVECAR")) == []
        assert list(root.rglob("CHGCAR")) == []

    def test_both_evaluations_are_stored(self, tmp_path):
        store, _root = self._run(tmp_path)

        assert {ev.structure_id for ev in store.get_evaluations("r")} == {
            "r/1",
            "r/2",
        }
