"""Add the evaluation.deviations column.



Revision ID: 0005
Revises: 0004
Create Date: 2026-09-18

"""
from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("evaluation", sa.Column("deviations", sa.Text()))


def downgrade() -> None:
    with op.batch_alter_table("evaluation") as batch_op:
        batch_op.drop_column("deviations")
