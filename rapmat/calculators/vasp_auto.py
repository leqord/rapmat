"""This module uses pymatgen, trying to match the Omat24 protocol (arXiv:2410.12771) as much as possible.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms

# ------------------------------------------------------------------ #
#  The OMat24 protocol
# ------------------------------------------------------------------ #

OMAT24_INCAR_OVERRIDES = {"ALGO": "Normal"}
OMAT24_POTCAR_FUNCTIONAL = "PBE_54"
OMAT24_POTCAR_VERSION = "54"
OMAT24_POTCAR_OVERRIDES = {"Yb": "Yb_3", "W": "W_sv"}

# NOTE: PBE POTCAR sets to look for, best first. "" is an unversioned directory
POTCAR_SET_PREFERENCE = ("54", "64", "52", "")

# NOTE: every VASP call should be a single point calculation
SINGLE_POINT_OVERRIDES = {
    "NSW": 0,
    "IBRION": -1,
    "LWAVE": False,
    "LCHARG": False,
}

VACUUM_AXIS = 2

_LDAU_KEYS = ("LDAUL", "LDAUU", "LDAUJ")


def pymatgen_version() -> str:
    from importlib.metadata import version

    return version("pymatgen")


def potcar_set_name(pp_version: str) -> str:
    return f"potpaw_PBE.{pp_version}" if pp_version else "potpaw_PBE"


def resolve_potcar_version() -> tuple[str | None, str | None]:
    from ase.config import cfg

    if "VASP_PP_VERSION" in cfg:
        return None, None

    root = cfg.get("VASP_PP_PATH")
    if not root:
        return None, None

    available = [
        version
        for version in POTCAR_SET_PREFERENCE
        if (Path(root) / potcar_set_name(version)).is_dir()
    ]
    if not available:
        return None, None

    chosen = available[0]
    if chosen == OMAT24_POTCAR_VERSION:
        return chosen, None

    note = (
        f"POTCAR set {potcar_set_name(chosen)}; OMat24 specifies "
        f"{potcar_set_name(OMAT24_POTCAR_VERSION)}"
    )
    return chosen, note


# ------------------------------------------------------------------ #
#  Generation
# ------------------------------------------------------------------ #


def _input_set(structure):
    from pymatgen.io.vasp.sets import BadInputSetWarning, MPRelaxSet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BadInputSetWarning)
        return MPRelaxSet(
            structure,
            sort_structure=False, # NOTE: important, MPRelaxSet would break MAGMOM otherwise
            user_incar_settings={
                **OMAT24_INCAR_OVERRIDES,
                **SINGLE_POINT_OVERRIDES,
            },
            user_potcar_functional=OMAT24_POTCAR_FUNCTIONAL,
            user_potcar_settings=dict(OMAT24_POTCAR_OVERRIDES),
        )


def omat24_vasp_params(
    atoms: "Atoms",
    *,
    monolayer: bool = False,
    potcar_version: str | None = OMAT24_POTCAR_VERSION,
) -> dict:
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = AseAtomsAdaptor.get_structure(atoms)
    input_set = _input_set(structure)
    incar = dict(input_set.incar)

    kpts = [int(k) for k in input_set.kpoints.kpts[0]]
    if monolayer:
        kpts[VACUUM_AXIS] = 1

    params = {
        key.lower(): value
        for key, value in incar.items()
        if key != "MAGMOM" and key not in _LDAU_KEYS
    }
    params["xc"] = "PBE"
    if potcar_version is not None:
        params["pp_version"] = potcar_version
    params["kpts"] = tuple(kpts)
    params["gamma"] = True

    # NOTE: seems working...
    if "MAGMOM" in incar:
        params["magmom"] = [float(m) for m in incar["MAGMOM"]]

    site_symbols = input_set.poscar.site_symbols

    ldau_luj = _ldau_luj(incar, site_symbols)
    if ldau_luj:
        params["ldau_luj"] = ldau_luj

    setups = _setups(site_symbols, input_set.potcar_symbols)
    if setups:
        params["setups"] = setups

    return params


def _ldau_luj(incar: dict, site_symbols: list[str]) -> dict:
    if not incar.get("LDAU"):
        return {}

    orbitals, u_values, j_values = (incar.get(key, []) for key in _LDAU_KEYS)
    return {
        symbol: {"L": int(orbital), "U": float(u), "J": float(j)}
        for symbol, orbital, u, j in zip(
            site_symbols, orbitals, u_values, j_values
        )
    }


def _setups(site_symbols: list[str], potcar_symbols: list[str]) -> dict:
    setups: dict[str, str] = {}
    for element, potcar in zip(site_symbols, potcar_symbols):
        suffix = potcar[len(element):]
        if suffix:
            setups[element] = suffix
    return setups


# ------------------------------------------------------------------ #
#  Reporting
# ------------------------------------------------------------------ #


def describe_params(params: dict) -> str:
    parts = []

    kpts = params.get("kpts")
    if kpts:
        parts.append("k-mesh " + "x".join(str(k) for k in kpts))

    for label, key in (("ENCUT", "encut"), ("ISPIN", "ispin")):
        if params.get(key) is not None:
            parts.append(f"{label} {params[key]:g}")

    ediff = params.get("ediff")
    if ediff is not None:
        parts.append(f"EDIFF {ediff:.2e}")

    setups = params.get("setups")
    if setups:
        parts.append(
            "POTCAR "
            + ", ".join(f"{el}{suffix}" for el, suffix in sorted(setups.items()))
        )

    hubbard = [
        f"{el}={luj['U']:g}"
        for el, luj in sorted((params.get("ldau_luj") or {}).items())
        if luj["U"]
    ]
    if hubbard:
        parts.append("U " + ", ".join(hubbard))

    return " | ".join(parts)


def export_toml(params: dict, label: str = "") -> str:
    import tomli_w

    subject = f" for {label}" if label else ""
    header = [
        f"# Auto-generated settings, based on the OMat24 protocol{subject}",
        f"# generated by rapmat from pymatgen {pymatgen_version()} MPRelaxSet",
        "# cell- and composition-specific. Regenerate per structure!",
    ]

    version = params.get("pp_version")
    if version is not None:
        line = f"# POTCAR set: {potcar_set_name(version)}"
        if version != OMAT24_POTCAR_VERSION:
            line += f" (OMat24 specifies {potcar_set_name(OMAT24_POTCAR_VERSION)})"
        header.append(line)

    header += ["", ""]
    return "\n".join(header) + tomli_w.dumps(_toml_safe(params))


def _toml_safe(params: dict) -> dict:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in params.items()
        if value is not None
    }
