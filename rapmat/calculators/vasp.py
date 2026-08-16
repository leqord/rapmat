from pathlib import Path

from ase.calculators.vasp import Vasp


def preflight_potcars(calculator, atoms) -> None:
    if not isinstance(calculator, Vasp):
        return

    try:
        calculator.initialize(atoms)
    except RuntimeError as exc:
        if "No pseudopotential" not in str(exc):
            return
        raise RuntimeError(
            f"{exc}\n"
            "ASE expects $VASP_PP_PATH/<set>/<symbol>/POTCAR."
            "A PBE run needs 'potpaw_PBE' or 'potpaw_PBE.<version>'."
        ) from exc


def build_calculator_vasp(config: dict, directory: Path | None = None) -> Vasp:
    kwargs = dict(config)

    if directory is not None and "directory" not in kwargs:
        kwargs["directory"] = str(directory)

    if "txt" not in kwargs:
        workdir = kwargs.get("directory", ".")
        kwargs["txt"] = str(Path(workdir) / "vasp.out")

    return Vasp(**kwargs)
