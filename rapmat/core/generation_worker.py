from rapmat.storage.status import StructureStatus


def generate_one_structure(
    struct_id: str,
    spg: int,
    fu: int,
    elements: list,
    formula_values: list,
    search_dim: int,
    thickness_cutoff: float | None,
    seed: int | None = None,
    max_count: int = 10,
) -> tuple:
    import pyxtal

    elements_number = [n * fu for n in formula_values]
    try:
        crystal = pyxtal.pyxtal()
        crystal.from_random(
            dim=search_dim,
            group=spg,
            species=elements,
            numIons=elements_number,
            max_count=max_count,
            thickness=thickness_cutoff if search_dim == 2 else None,
            random_state=seed,
        )
        if crystal.valid:
            atoms = crystal.to_ase()

            return (StructureStatus.GENERATED, struct_id, atoms)
        return (StructureStatus.DISCARDED, struct_id, None)
    except pyxtal.msg.Comp_CompatibilityError:
        return (StructureStatus.DISCARDED, struct_id, None)
    except RuntimeError:
        return (StructureStatus.ERROR, struct_id, None)
