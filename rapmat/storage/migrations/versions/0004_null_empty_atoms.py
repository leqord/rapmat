"""Store NULL instead of '' for missing atoms payloads.

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-06

"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "UPDATE structure SET initial_atoms_json = NULL WHERE initial_atoms_json = ''"
    )
    op.execute(
        "UPDATE structure SET final_atoms_json = NULL WHERE final_atoms_json = ''"
    )


def downgrade() -> None:
    pass
