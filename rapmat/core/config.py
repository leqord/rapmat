"""Typed search/run configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


def merge_config_dicts(
    study_cfg: dict,
    batch_cfg: dict,
    *,
    system: str = "",
    domain: str = "",
    calculator: str = "",
) -> dict:
    """Merge a study config with a run's batch config.
    """
    merged = {**study_cfg, **batch_cfg}
    if domain:
        merged["domain"] = domain
    if system:
        merged["system"] = system
    if calculator:
        merged["calculator"] = calculator
    return merged


class SearchConfig(BaseModel):
    """Effective configuration for a search/run.

    ``extra="ignore"`` allows old records with retired keys (e.g. ``dedup``,
    ``dedup_threshold``) validate without error.
    """

    model_config = ConfigDict(extra="ignore")

    # Run
    formula: dict[str, int] = Field(default_factory=dict)
    formula_units: list[int] = Field(default_factory=lambda: [2, 4])
    candidates_per_group: int = 2
    seed: int | None = None

    # Study columns
    system: str = ""
    domain: str = "bulk"
    calculator: str = "MATTERSIM"

    # Method defaults
    calculator_config: dict = Field(default_factory=dict)
    symprec: float = 1e-2
    pressure_gpa: float = 0.0
    min_dist: float = 0.5
    sanity_pymatgen: bool = False
    sanity_pymatgen_tol: float = 0.5
    force_conv_crit: float = 5e-3
    steps_max: int = 2000
    forces_break: float = 1000.0
    thickness_cutoff: float | None = None
    max_count: int = 10
    phonon_cutoff: float | None = None

    @classmethod
    def from_stored(
        cls,
        study_cfg: dict,
        batch_cfg: dict,
        *,
        system: str = "",
        domain: str = "",
        calculator: str = "",
    ) -> "SearchConfig":
        """Build the effective config from the two stored config dicts."""
        return cls.model_validate(
            merge_config_dicts(
                study_cfg,
                batch_cfg,
                system=system,
                domain=domain,
                calculator=calculator,
            )
        )
