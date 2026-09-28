import json
from collections import Counter
from typing import TYPE_CHECKING, Iterable, Literal, Sequence

from pydantic import (BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt,
                      ValidationError)

from rapmat.calculators import REQUIRES_EXTERNAL_CONFIG, Calculators

if TYPE_CHECKING:
    from rapmat.storage.base import StructureStore

DEFAULT_PHONON_CUTOFF = -0.15

_MACHINE_KEYS = frozenset({"command", "directory"})

_Grid = tuple[PositiveInt, PositiveInt, PositiveInt]


class PhononSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    calculator: Calculators
    calculator_settings: Literal["toml", "auto"]
    calculator_config: dict = Field(default_factory=dict)
    supercell: _Grid
    mesh: _Grid
    displacement: PositiveFloat
    symprec: PositiveFloat
    reduce_primitive: bool
    config_path: str = ""

    @classmethod
    def from_json(cls, text: str | None) -> "PhononSettings | None":
        if not text:
            return None
        try:
            return cls.model_validate_json(text)
        except ValidationError:
            return None

    def _identity_dict(self) -> dict:
        data = self.model_dump(mode="json", exclude={"config_path"})
        data["calculator_config"] = {
            k: v for k, v in data["calculator_config"].items() if k not in _MACHINE_KEYS
        }
        return data

    def identity(self) -> str:
        return json.dumps(self._identity_dict(), sort_keys=True)

    def differences(self, other: "PhononSettings") -> list[str]:
        mine, theirs = self._identity_dict(), other._identity_dict()
        return [key for key in mine if mine[key] != theirs.get(key)]

    def run_config(self, command: str = "") -> dict:
        config = dict(self.calculator_config)
        if command and self.calculator in REQUIRES_EXTERNAL_CONFIG:
            config["command"] = command
        return config


def default_phonon_grid(domain: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if domain == "monolayer":
        return (3, 3, 1), (20, 20, 1)
    return (3, 3, 3), (20, 20, 20)


StoredPhonons = dict[str, tuple[str | None, "PhononSettings | None"]]


def load_stored_phonons(
    store: "StructureStore", run_names: Sequence[str]
) -> StoredPhonons:
    return {
        sid: (run, PhononSettings.from_json(text))
        for sid, (run, text) in store.get_phonon_settings(run_names).items()
    }


def split_stored(
    stored: StoredPhonons, settings: PhononSettings
) -> tuple[set[str], dict[str, str | None]]:
    identity = settings.identity()
    matching: set[str] = set()
    stale: dict[str, str | None] = {}
    for sid, (run, known) in stored.items():
        if known is not None and known.identity() == identity:
            matching.add(sid)
        else:
            stale[sid] = run
    return matching, stale


def stale_differences(
    stored: StoredPhonons, stale_ids: Iterable[str], settings: PhononSettings
) -> list[str]:
    fields: list[str] = []
    unrecorded = False
    for sid in stale_ids:
        known = stored[sid][1]
        if known is None:
            unrecorded = True
            continue
        for name in known.differences(settings):
            if name not in fields:
                fields.append(name)
    if unrecorded:
        fields.append("not recorded")
    return fields


def prevailing_settings(stored: StoredPhonons) -> PhononSettings | None:
    known = [s for _run, s in stored.values() if s is not None]
    if not known:
        return None
    identity, _count = Counter(s.identity() for s in known).most_common(1)[0]
    candidates = [s for s in known if s.identity() == identity]
    return next((s for s in candidates if s.config_path), candidates[0])


def resolve_phonon_cutoff(
    store: "StructureStore", study_id: str | None, run_names: Sequence[str]
) -> float:
    if study_id:
        study = store.get_study(study_id)
        value = (study.config or {}).get("phonon_cutoff") if study else None
        if value is not None:
            return float(value)
        runs = [m for m in store.get_study_runs(study_id) if m.name in run_names]
    else:
        runs = [m for m in (store.get_run_metadata(n) for n in run_names) if m]

    values = {
        float(v) for m in runs if (v := m.config.get("phonon_cutoff")) is not None
    }
    return values.pop() if len(values) == 1 else DEFAULT_PHONON_CUTOFF


def save_phonon_cutoff(
    store: "StructureStore",
    study_id: str | None,
    run_names: Sequence[str],
    value: float,
) -> None:
    if not study_id:
        for name in run_names:
            store.set_run_config_value(name, "phonon_cutoff", value)
        return

    store.set_study_config_value(study_id, "phonon_cutoff", value)
    for meta in store.get_study_runs(study_id):
        if "phonon_cutoff" in meta.config:
            config = dict(meta.config)
            config.pop("phonon_cutoff")
            store.update_run_config(meta.name, config)
