"""Add the structure.excluded flag.



Revision ID: 0002
Revises: 0001
Create Date: 2026-07-03

"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("structure", sa.Column("excluded", sa.Integer()))


def downgrade() -> None:
    with op.batch_alter_table("structure") as batch_op:
        batch_op.drop_column("excluded")
