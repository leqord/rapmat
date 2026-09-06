from pathlib import Path

from ase.calculators.vasp import Vasp


class RapmatVasp(Vasp):

    def write_input(self, atoms, properties=None, system_changes=None):
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        super().write_input(atoms, properties, system_changes)


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

    # NOTE: important
    kwargs.setdefault("txt", "vasp.out")

    return RapmatVasp(**kwargs)
