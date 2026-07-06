"""Shared (de)serialization helpers for storage backends.
"""

_decode = None
_encode = None


def ase_decode(raw):
    global _decode
    if _decode is None:
        from ase.io.jsonio import decode
        _decode = decode
    return _decode(raw)


def ase_encode(atoms):
    global _encode
    if _encode is None:
        from ase.io.jsonio import encode
        _encode = encode
    return _encode(atoms)
