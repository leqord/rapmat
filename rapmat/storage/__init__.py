def __getattr__(name: str):
    if name == "SOAPDescriptor":
        from rapmat.storage.descriptors import SOAPDescriptor

        return SOAPDescriptor
    if name == "SQLiteStore":
        from rapmat.storage.sqlite_store import SQLiteStore

        return SQLiteStore

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
