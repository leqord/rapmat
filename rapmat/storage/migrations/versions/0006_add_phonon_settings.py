"""Add the phonon.settings_json column.



Revision ID: 0006
Revises: 0005
Create Date: 2026-09-29

"""
from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("phonon", sa.Column("settings_json", sa.Text()))


def downgrade() -> None:
    with op.batch_alter_table("phonon") as batch_op:
        batch_op.drop_column("settings_json")
