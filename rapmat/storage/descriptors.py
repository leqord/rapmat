from typing import List

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP


class SOAPDescriptor:
    """Computes averaged SOAP vectors."""

    def __init__(
        self,
        species: List[str],
        r_cut: float = 6.0,
        n_max: int = 8,
        l_max: int = 6,
        periodic: bool = True,
    ):
        self._soap = SOAP(
            species=species,
            periodic=periodic,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            average="inner",
            sparse=False,
        )
        self._dim = self._soap.get_number_of_features()

    def dimension(self) -> int:
        return self._dim

    def compute(self, atoms: Atoms) -> np.ndarray:
        vec = self._soap.create(atoms)
        return vec.flatten()
