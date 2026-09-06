import json
from typing import Optional

from ase import Atoms
from sqlalchemy import Integer, Text
from sqlalchemy.types import TypeDecorator

from rapmat.storage._serde import ase_decode, ase_encode


class AtomsJSON(TypeDecorator):
    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Optional[Atoms], dialect) -> Optional[str]:
        return ase_encode(value) if value is not None else None

    def process_result_value(self, value: Optional[str], dialect) -> Optional[Atoms]:
        return ase_decode(value) if value else None


class JSONDict(TypeDecorator):
    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Optional[dict], dialect) -> str:
        return json.dumps(value if value is not None else {})

    def process_result_value(self, value: Optional[str], dialect) -> dict:
        return json.loads(value) if value else {}


class IntBool(TypeDecorator):
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value, dialect) -> int:
        return 1 if value else 0

    def process_result_value(self, value, dialect) -> bool:
        return bool(value or 0)


class OptIntBool(TypeDecorator):
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value, dialect) -> Optional[int]:
        return None if value is None else (1 if value else 0)

    def process_result_value(self, value, dialect) -> Optional[bool]:
        return None if value is None else bool(value)
