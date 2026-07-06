"""Create real foreign keys with ON DELETE CASCADE.



Revision ID: 0003
Revises: 0002
Create Date: 2026-07-03

"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_FKS = [
    ("run", "fk_run_study_study", "study", "study", "id"),
    ("structure", "fk_structure_run_run", "run", "run", "name"),
    ("evaluation", "fk_evaluation_structure_structure", "structure", "structure", "id"),
    ("evaluation", "fk_evaluation_run_run", "run", "run", "name"),
    ("phonon", "fk_phonon_structure_structure", "structure", "structure", "id"),
    ("phonon", "fk_phonon_run_run", "run", "run", "name"),
    ("phonon_params", "fk_phonon_params_structure_structure", "structure", "structure", "id"),
    ("phonon_params", "fk_phonon_params_run_run", "run", "run", "name"),
]


def upgrade() -> None:
    for child in ("evaluation", "phonon", "phonon_params"):
        op.execute(
            f"DELETE FROM {child} WHERE "
            "(run IS NOT NULL AND run NOT IN (SELECT name FROM run)) "
            "OR (structure IS NOT NULL "
            "AND structure NOT IN (SELECT id FROM structure))"
        )
    op.execute(
        "DELETE FROM structure WHERE "
        "run IS NOT NULL AND run NOT IN (SELECT name FROM run)"
    )
    op.execute(
        "DELETE FROM run WHERE "
        "study IS NOT NULL AND study NOT IN (SELECT id FROM study)"
    )

    op.execute("UPDATE structure SET excluded = 0 WHERE excluded IS NULL")
    op.execute("UPDATE structure SET converged = 0 WHERE converged IS NULL")

    tables: dict[str, list] = {}
    for table, name, ref, local, remote in _FKS:
        tables.setdefault(table, []).append((name, ref, local, remote))

    for table, fks in tables.items():
        with op.batch_alter_table(table) as batch_op:
            for name, ref, local, remote in fks:
                batch_op.create_foreign_key(
                    name, ref, [local], [remote], ondelete="CASCADE"
                )


def downgrade() -> None:
    tables: dict[str, list] = {}
    for table, name, *_ in _FKS:
        tables.setdefault(table, []).append(name)

    for table, names in tables.items():
        with op.batch_alter_table(table) as batch_op:
            for name in names:
                batch_op.drop_constraint(name, type_="foreignkey")
