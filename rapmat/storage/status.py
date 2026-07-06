from enum import StrEnum


class RunStatus(StrEnum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class StructureStatus(StrEnum):
    GENERATING = "generating"
    GENERATED = "generated"
    RELAXED = "relaxed"
    DISCARDED = "discarded"
    ERROR = "error"
