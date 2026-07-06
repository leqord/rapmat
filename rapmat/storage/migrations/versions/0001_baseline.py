"""Baseline.

Existing databases are here (or at 0002 if they already have
`excluded`).

Revision ID: 0001
Revises:
Create Date: 2026-07-03

"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "study",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("system", sa.Text()),
        sa.Column("domain", sa.Text()),
        sa.Column("calculator", sa.Text()),
        sa.Column("config_json", sa.Text()),
        sa.Column("timestamp", sa.Text()),
    )
    op.create_table(
        "run",
        sa.Column("name", sa.Text(), primary_key=True),
        sa.Column("batch_config_json", sa.Text()),
        sa.Column("timestamp", sa.Text()),
        sa.Column("study", sa.Text()),
        sa.Column("run_status", sa.Text()),
        sa.Column("worker_id", sa.Text()),
        sa.Column("heartbeat", sa.Text()),
    )
    op.create_table(
        "structure",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("run", sa.Text()),
        sa.Column("status", sa.Text()),
        sa.Column("gen_spg", sa.Integer()),
        sa.Column("gen_fu", sa.Integer()),
        sa.Column("energy_per_atom", sa.Float()),
        sa.Column("fmax", sa.Float()),
        sa.Column("converged", sa.Integer()),
        sa.Column("duplicate", sa.Integer()),
        sa.Column("initial_atoms_json", sa.Text()),
        sa.Column("final_atoms_json", sa.Text()),
    )
    op.create_table(
        "evaluation",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("structure", sa.Text()),
        sa.Column("run", sa.Text()),
        sa.Column("calculator", sa.Text()),
        sa.Column("config_json", sa.Text()),
        sa.Column("energy_per_atom", sa.Float()),
        sa.Column("energy_total", sa.Float()),
        sa.Column("min_phonon_freq", sa.Float()),
    )
    op.create_table(
        "phonon",
        sa.Column("structure", sa.Text(), primary_key=True),
        sa.Column("run", sa.Text()),
        sa.Column("min_phonon_freq", sa.Float()),
        sa.Column("supercell", sa.Text()),
        sa.Column("mesh", sa.Text()),
        sa.Column("displacement", sa.Float()),
        sa.Column("symprec", sa.Float()),
        sa.Column("calculator", sa.Text()),
    )
    op.create_table(
        "phonon_params",
        sa.Column("structure", sa.Text(), primary_key=True),
        sa.Column("run", sa.Text()),
        sa.Column("params_gz", sa.Text()),
    )

    op.create_index("idx_struct_run", "structure", ["run"])
    op.create_index("idx_struct_status", "structure", ["status"])
    op.create_index("idx_struct_run_status", "structure", ["run", "status"])
    op.create_index("idx_eval_run", "evaluation", ["run"])
    op.create_index("idx_eval_struct", "evaluation", ["structure"])
    op.create_index("idx_phonon_run", "phonon", ["run"])
    op.create_index("idx_phonon_params_run", "phonon_params", ["run"])


def downgrade() -> None:
    for table in (
        "phonon_params",
        "phonon",
        "evaluation",
        "structure",
        "run",
        "study",
    ):
        op.drop_table(table)
