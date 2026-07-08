import importlib.util
import subprocess
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable


class Calculators(str, Enum):
    MATTERSIM = "MATTERSIM"
    NEQUIP_OAML = "NEQUIP-OAML"
    UPET = "UPET"
    VASP = "VASP"


CALCULATOR_META: dict[Calculators, dict] = {
    Calculators.MATTERSIM: {
        "probe": "mattersim.forcefield",
        "extra": "mattersim",
        "description": "MatterSim 5M",
    },
    Calculators.NEQUIP_OAML: {
        "probe": "nequip.ase",
        "extra": "nequip",
        "description": "NequIP OAM-L",
    },
    Calculators.UPET: {
        "probe": "upet.calculator",
        "extra": "upet",
        "description": "UPET OAM-XL",
    },
    Calculators.VASP: {
        "probe": "ase.calculators.vasp",
        "extra": None,
        "description": "VASP",
    },
}


REQUIRES_EXTERNAL_CONFIG: frozenset[Calculators] = frozenset({
    Calculators.VASP,
})


def probe_calculator(calc: Calculators) -> tuple[bool, str | None]:
    """(available, import error).

    A package that is installed but fails to import 
    counts as unavailable, with the error message returned.
    """
    try:
        found = importlib.util.find_spec(CALCULATOR_META[calc]["probe"]) is not None
        return found, None
    except (ModuleNotFoundError, ValueError):
        return False, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def is_calculator_available(calc: Calculators) -> bool:
    return probe_calculator(calc)[0]


def get_install_hint(calc: Calculators) -> str | None:
    extra = CALCULATOR_META[calc]["extra"]
    return f"pip install rapmat[{extra}]" if extra else None


# ------------------------------------------------------------------ #
#  Calculator loading callback protocol
# ------------------------------------------------------------------ #


@runtime_checkable
class CalculatorCallback(Protocol):
    def on_status(self, message: str) -> None: ...


class ProgressCalcCallback:
    def __init__(self, progress_callback) -> None:
        self._progress_callback = progress_callback

    def on_status(self, message: str) -> None:
        if self._progress_callback:
            self._progress_callback(0, 0, message)


class LogCalcCallback:
    def __init__(self, log_fn) -> None:
        self._log_fn = log_fn

    def on_status(self, message: str) -> None:
        if self._log_fn:
            self._log_fn(message)


def _notify(callback: CalculatorCallback | None, message: str) -> None:
    if callback is not None:
        callback.on_status(message)


def ensure_asset(
    name: str,
    path: Path,
    install_fn: Callable[[], subprocess.CompletedProcess],
    callback: CalculatorCallback | None = None,
    log_path: Path | None = None,
) -> None:
    if path.exists():
        return

    _notify(callback, f"{name} not found at {path}, installing...")

    try:
        result = install_fn()
    except Exception as e:
        raise RuntimeError(f"Failed to install {name}: {e}") from e

    if result.returncode != 0:
        stderr = result.stderr.decode() if result.stderr else ""
        log_msg = f"\nSee log file for details: {log_path}" if log_path else ""
        err_msg = stderr or "No stderr output available."
        raise RuntimeError(
            f"Failed to install {name} (exit code {result.returncode}):\n{err_msg}{log_msg}"
        )

    if not path.exists():
        raise RuntimeError(
            f"Install command succeeded but {path} still does not exist."
        )

    _notify(callback, f"{name} installed successfully.")


def cleanup_calculator_files(calculator) -> None:
    calc_name = getattr(calculator, "name", "").lower()

    if calc_name == "vasp" and hasattr(calculator, "directory"):
        for fname in [
            "WAVECAR",
            "WAVECAR.h5",
            "CHGCAR",
            "CHG",
            "vasprun.xml",
            "OUTCAR",
            "OSZICAR",
            "EIGENVAL",
            "DOSCAR",
            "PROCAR",
            "IBZKPT",
            "PCDAT",
            "XDATCAR",
            "CONTCAR",
            "vasp*.lock",
        ]:
            fpath = Path(calculator.directory) / fname
            if fpath.exists():
                try:
                    fpath.unlink()
                except OSError:
                    pass
