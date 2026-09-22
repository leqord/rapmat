from pathlib import Path

from rapmat.calculators import (CalculatorCallback, Calculators,
                                get_install_hint, is_calculator_available)
from rapmat.calculators.calc_dirs import CalcDirAllocator


class CalculatorProvider:
    def __init__(
        self,
        calculator_name: Calculators,
        output_dir_path: Path | None = None,
        config: dict | None = None,
        callback: CalculatorCallback | None = None,
        *,
        auto_settings: bool = False,
        monolayer: bool = False,
        log_callback=None,
    ) -> None:
        self._name = Calculators(calculator_name)
        self._output_dir_path = output_dir_path
        self._config = dict(config or {})
        self._callback = callback
        self._auto = auto_settings and self._name is Calculators.VASP
        self._monolayer = monolayer
        self._log_callback = log_callback
        self._cached = None
        self._potcar_version = None
        self._overrides: dict = {}
        self._run_deviations: dict = {}
        self._label_seen = None

        # NOTE: Only VASP writes to disk
        self._dirs = CalcDirAllocator(
            output_dir_path if self._name is Calculators.VASP else None
        )

        if self._dirs.active and self._config.pop("directory", None) is not None:
            if log_callback:
                log_callback(
                    "Ignoring 'directory' from the calculator config: "
                    "one directory per calculation."
                )

        if self._auto:
            from rapmat.calculators.vasp_auto import (MONOLAYER_ISMEAR,
                                                      monolayer_ismear_notice,
                                                      resolve_potcar_version)

            self._potcar_version, note = resolve_potcar_version()
            if note and log_callback:
                log_callback(note)

            if self._monolayer:
                self._run_deviations = {"ismear": MONOLAYER_ISMEAR}
                from rapmat.calculators.vasp_recovery import warn

                warn(log_callback, monolayer_ismear_notice())

    @property
    def auto(self) -> bool:
        return self._auto

    @property
    def scope_path(self) -> Path | None:
        return self._dirs.scope_path

    def set_calc_label(self, label) -> None:
        # NOTE: label has to be normalised exactly as the allocator does this
        normalized = None if label is None else str(label)
        if normalized != self._label_seen:
            self._overrides = {}
        self._label_seen = normalized
        self._dirs.set_label(label)

    def finalize(self) -> None:
        self._dirs.finalize()

    def apply_overrides(self, patch: dict) -> None:
        from rapmat.calculators.vasp_recovery import RECOVERY_KEYS

        forbidden = set(patch) - RECOVERY_KEYS
        if forbidden:
            raise ValueError(
                f"recovery may not change {sorted(forbidden)}; "
                f"only {sorted(RECOVERY_KEYS)} are recoverable"
            )
        self._overrides.update(patch)

    @property
    def overrides(self) -> dict:
        return dict(self._overrides)

    @property
    def deviations(self) -> dict:
        return {**self._run_deviations, **self._overrides}

    @property
    def pinned_recovery_keys(self) -> frozenset:
        from rapmat.calculators.vasp_recovery import RECOVERY_KEYS

        return frozenset(key for key in RECOVERY_KEYS if key in self._config)

    @property
    def last_directory(self):
        return self._dirs.current

    def __call__(self, atoms):
        directory = self._dirs.next()

        if not self._auto:
            if self._overrides:
                # NOTE: do not mutate the shared instance
                return self._build({**self._config, **self._overrides}, directory)
            if self._cached is None:
                self._cached = self._build(self._config, directory)
            elif directory is not None:
                self._cached.directory = str(directory)
            return self._cached

        from rapmat.calculators.vasp_auto import (describe_params,
                                                  omat24_vasp_params)

        params = omat24_vasp_params(
            atoms,
            monolayer=self._monolayer,
            potcar_version=self._potcar_version,
        )
        merged = {**self._config, **params, **self._overrides}
        if self._log_callback:
            self._log_callback(
                f"{atoms.get_chemical_formula()}: {describe_params(merged)}"
            )
        return self._build(merged, directory)

    def reset(self) -> None:
        self._cached = None

    def _build(self, config: dict, directory: Path | None = None):
        return load_calculator(
            self._name,
            directory if directory is not None else self._output_dir_path,
            config=config,
            callback=self._callback,
        )


def set_provider_label(calculator_for, label) -> None:
    setter = getattr(calculator_for, "set_calc_label", None)
    if callable(setter):
        setter(label)


def finalize_provider(calculator_for) -> None:
    finalizer = getattr(calculator_for, "finalize", None)
    if callable(finalizer):
        finalizer()


def apply_provider_overrides(calculator_for, patch: dict) -> bool:
    applier = getattr(calculator_for, "apply_overrides", None)
    if not callable(applier):
        return False
    applier(patch)
    return True


def provider_overrides(calculator_for) -> dict:
    return dict(getattr(calculator_for, "overrides", None) or {})


def provider_deviations(calculator_for) -> dict:
    return dict(getattr(calculator_for, "deviations", None) or {})


def provider_pinned_keys(calculator_for) -> frozenset:
    return frozenset(getattr(calculator_for, "pinned_recovery_keys", None) or ())


def provider_last_directory(calculator_for):
    return getattr(calculator_for, "last_directory", None)


def load_calculator(
    calculator_name: Calculators,
    output_dir_path: Path | None = None,
    config: dict | None = None,
    callback: CalculatorCallback | None = None,
):
    try:
        match calculator_name.value:
            case Calculators.MATTERSIM.value:
                from rapmat.calculators.mattersim import \
                    build_calculator_mattersim

                return build_calculator_mattersim()
            case Calculators.NEQUIP_OAML.value:
                from rapmat.calculators.nequip import \
                    build_calculator_nequip_oaml

                return build_calculator_nequip_oaml(callback=callback)
            case Calculators.UPET.value:
                from rapmat.calculators.upet import build_calculator_upet

                return build_calculator_upet(config, callback=callback)
            case Calculators.VASP.value:
                from rapmat.calculators.vasp import build_calculator_vasp

                return build_calculator_vasp(config or {}, output_dir_path)
            case _:
                raise NotImplementedError(
                    f"Calculator {calculator_name.value} is not implemented"
                )
    except ImportError as ie:
        hint = get_install_hint(calculator_name)
        installed = [c.value for c in Calculators if is_calculator_available(c)]
        msg = f"Calculator {calculator_name.value} is not installed."
        if hint:
            msg += f"\n  Install with: {hint}"
        if installed:
            msg += f"\n  Currently available: {', '.join(installed)}"
        raise ImportError(msg) from ie
    except RuntimeError as re:
        raise RuntimeError(
            f"Failed to initialize {calculator_name.value}: {re}"
        ) from re
