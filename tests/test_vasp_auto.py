"""Tests for the per-structure VASP settings auto-generator.
"""

import tomllib

import pytest
from ase import Atoms
from ase.build import bulk, mx2
from ase.calculators.vasp import Vasp

from rapmat.calculators.vasp_auto import (OMAT24_POTCAR_VERSION,
                                          describe_params, export_toml,
                                          omat24_vasp_params)
from rapmat.calculators.vasp import build_calculator_vasp


@pytest.fixture
def si():
    return bulk("Si", "diamond", a=5.43)


@pytest.fixture
def fe():
    return bulk("Fe", "bcc", a=2.87)


@pytest.fixture
def fe_oxide_interleaved():
    return Atoms(
        symbols="FeOFeOO",
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25],
        ],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )


# ------------------------------------------------------------------ #
#  The OMat24 protocol
# ------------------------------------------------------------------ #


class TestOmat24Protocol:
    def test_algo_normal(self, si):
        assert omat24_vasp_params(si)["algo"] == "Normal"

    def test_single_point_overrides(self, si):
        params = omat24_vasp_params(si)
        assert params["nsw"] == 0
        assert params["ibrion"] == -1
        assert params["lwave"] is False
        assert params["lcharg"] is False

    def test_potcar_version_54(self, si):
        params = omat24_vasp_params(si)
        assert params["xc"] == "PBE"
        assert params["pp_version"] == OMAT24_POTCAR_VERSION == "54"

    def test_mprelaxset_defaults_carried_through(self, si):
        params = omat24_vasp_params(si)
        assert params["encut"] == pytest.approx(520.0)
        assert params["prec"] == "Accurate"
        assert params["ispin"] == 2

    def test_ediff_scales_with_atom_count(self, si):
        small = omat24_vasp_params(si)["ediff"]
        large = omat24_vasp_params(si.repeat((2, 1, 1)))["ediff"]
        assert large > small


# ------------------------------------------------------------------ #
#  k-mesh
# ------------------------------------------------------------------ #


class TestKpoints:
    def test_gamma_centred(self, si):
        assert omat24_vasp_params(si)["gamma"] is True

    def test_mesh_scales_with_cell(self, si):
        unit = omat24_vasp_params(si)["kpts"]
        super_ = omat24_vasp_params(si.repeat((2, 2, 2)))["kpts"]
        assert all(s <= u for s, u in zip(super_, unit))
        assert super_ != unit

    def test_monolayer_pins_vacuum_axis(self):
        atoms = mx2("MoS2", vacuum=10.0)
        assert omat24_vasp_params(atoms, monolayer=True)["kpts"][2] == 1

    def test_monolayer_leaves_in_plane_mesh_alone(self):
        atoms = mx2("MoS2", vacuum=10.0)
        bulk_kpts = omat24_vasp_params(atoms, monolayer=False)["kpts"]
        mono_kpts = omat24_vasp_params(atoms, monolayer=True)["kpts"]
        assert mono_kpts[:2] == bulk_kpts[:2]

    def test_monolayer_pin_overrides_a_thin_vacuum(self):
        atoms = mx2("MoS2", vacuum=1.0)
        assert omat24_vasp_params(atoms, monolayer=False)["kpts"][2] > 1
        assert omat24_vasp_params(atoms, monolayer=True)["kpts"][2] == 1


# ------------------------------------------------------------------ #
#  MAGMOM ordering and dtype
# ------------------------------------------------------------------ #


class TestMagmom:
    def test_one_value_per_atom(self, fe_oxide_interleaved):
        params = omat24_vasp_params(fe_oxide_interleaved)
        assert len(params["magmom"]) == len(fe_oxide_interleaved)

    def test_follows_atoms_order_not_species_order(self, fe_oxide_interleaved):
        params = omat24_vasp_params(fe_oxide_interleaved)
        by_symbol = list(
            zip(fe_oxide_interleaved.get_chemical_symbols(), params["magmom"])
        )
        assert [m for s, m in by_symbol if s == "Fe"] == [5.0, 5.0]
        assert [m for s, m in by_symbol if s == "O"] == [0.6, 0.6, 0.6]

    def test_values_are_floats(self, fe_oxide_interleaved):
        params = omat24_vasp_params(fe_oxide_interleaved)
        assert all(isinstance(m, float) for m in params["magmom"])

    def test_survives_ase_sorting(self, fe_oxide_interleaved):
        params = omat24_vasp_params(fe_oxide_interleaved)
        sort, _ = Vasp(**params)._make_sort(fe_oxide_interleaved)

        symbols = fe_oxide_interleaved.get_chemical_symbols()
        for index in sort:
            expected = 5.0 if symbols[index] == "Fe" else 0.6
            assert params["magmom"][index] == expected


# ------------------------------------------------------------------ #
#  Hubbard U
# ------------------------------------------------------------------ #


class TestHubbardU:
    def test_applied_to_oxide(self, fe_oxide_interleaved):
        ldau = omat24_vasp_params(fe_oxide_interleaved)["ldau_luj"]
        assert ldau["Fe"] == {"L": 2, "U": pytest.approx(5.3), "J": 0.0}

    def test_not_applied_to_the_anion(self, fe_oxide_interleaved):
        ldau = omat24_vasp_params(fe_oxide_interleaved)["ldau_luj"]
        assert ldau["O"]["U"] == 0.0
        assert ldau["O"]["L"] == -1 or ldau["O"]["L"] == 0

    def test_absent_for_a_plain_metal(self, fe):
        assert "ldau_luj" not in omat24_vasp_params(fe)

    def test_absent_for_a_non_u_element(self, si):
        assert "ldau_luj" not in omat24_vasp_params(si)

    def test_uses_dict_not_positional_lists(self, fe_oxide_interleaved):
        params = omat24_vasp_params(fe_oxide_interleaved)
        assert "ldaul" not in params
        assert "ldauu" not in params
        assert "ldauj" not in params


# ------------------------------------------------------------------ #
#  POTCAR setups
# ------------------------------------------------------------------ #


class TestSetups:
    def test_mp_recommended_setup(self, fe):
        assert omat24_vasp_params(fe)["setups"] == {"Fe": "_pv"}

    def test_omitted_when_plain(self, si):
        assert "setups" not in omat24_vasp_params(si)

    def test_multi_element(self):
        params = omat24_vasp_params(mx2("MoS2", vacuum=10.0))
        assert params["setups"] == {"Mo": "_pv"}

    def test_omat24_w_override(self):
        atoms = bulk("W", "bcc", a=3.16)
        assert omat24_vasp_params(atoms)["setups"]["W"] == "_sv"


# ------------------------------------------------------------------ #
#  ASE acceptance
# ------------------------------------------------------------------ #


class TestAseAcceptance:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: bulk("Si", "diamond", a=5.43),
            lambda: bulk("Fe", "bcc", a=2.87),
            lambda: mx2("MoS2", vacuum=10.0),
        ],
        ids=["Si", "Fe", "MoS2"],
    )
    def test_params_construct_a_calculator(self, factory):
        assert isinstance(Vasp(**omat24_vasp_params(factory())), Vasp)

    def test_oxide_constructs(self, fe_oxide_interleaved):
        params = omat24_vasp_params(fe_oxide_interleaved)
        assert isinstance(Vasp(**params), Vasp)

    def test_goes_through_build_calculator_vasp(self, si, tmp_path):
        params = omat24_vasp_params(si)
        params["command"] = "mpirun -np 4 vasp_std"
        calc = build_calculator_vasp(params, directory=tmp_path)
        assert isinstance(calc, Vasp)
        assert calc.directory == str(tmp_path)


# ------------------------------------------------------------------ #
#  Reporting
# ------------------------------------------------------------------ #


class TestDescribeParams:
    def test_mentions_mesh_and_cutoff(self, si):
        text = describe_params(omat24_vasp_params(si))
        assert "7x7x7" in text
        assert "ENCUT 520" in text

    def test_reports_hubbard_u(self, fe_oxide_interleaved):
        text = describe_params(omat24_vasp_params(fe_oxide_interleaved))
        assert "Fe=5.3" in text

    def test_omits_u_when_absent(self, si):
        assert "U " not in describe_params(omat24_vasp_params(si))

    def test_reports_potcar_setup(self, fe):
        assert "Fe_pv" in describe_params(omat24_vasp_params(fe))


class TestExportToml:
    def test_header_records_provenance(self, si):
        text = export_toml(omat24_vasp_params(si), "Si2")
        assert "Si2" in text
        assert "pymatgen" in text
        assert "OMat24" in text

    def test_header_records_the_potcar_set(self, si):
        text = export_toml(omat24_vasp_params(si))
        assert "POTCAR set: potpaw_PBE.54" in text

    def test_header_flags_a_non_omat24_potcar_set(self, si):
        text = export_toml(omat24_vasp_params(si, potcar_version=""))
        assert "POTCAR set: potpaw_PBE " in text
        assert "OMat24 specifies potpaw_PBE.54" in text

    def test_is_valid_toml(self, si):
        loaded = tomllib.loads(export_toml(omat24_vasp_params(si)))
        assert loaded["algo"] == "Normal"
        assert loaded["kpts"] == [7, 7, 7]
        assert loaded["gamma"] is True

    def test_uses_gamma_not_kgamma(self, si):
        loaded = tomllib.loads(export_toml(omat24_vasp_params(si)))
        assert "kgamma" not in loaded

    def test_round_trips_through_the_toml_loader(self, fe_oxide_interleaved):
        params = omat24_vasp_params(fe_oxide_interleaved)
        loaded = tomllib.loads(export_toml(params))

        expected = {
            k: list(v) if isinstance(v, tuple) else v for k, v in params.items()
        }
        assert loaded == expected

    def test_round_trip_rebuilds_the_same_calculator(self, fe_oxide_interleaved):
        params = omat24_vasp_params(fe_oxide_interleaved)
        loaded = tomllib.loads(export_toml(params))

        direct = Vasp(**params)
        restored = build_calculator_vasp(loaded)

        assert restored.string_params["algo"] == direct.string_params["algo"]
        assert restored.float_params["encut"] == direct.float_params["encut"]
        assert restored.dict_params["ldau_luj"] == direct.dict_params["ldau_luj"]
        assert (
            restored.list_float_params["magmom"]
            == direct.list_float_params["magmom"]
        )
        assert (
            tuple(restored.input_params["kpts"])
            == tuple(direct.input_params["kpts"])
        )
