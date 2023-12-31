"""Added Image Table

Revision ID: 4796996bfcbe
Revises: 
Create Date: 2023-08-11 00:36:38.550908

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4796996bfcbe'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('Image',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('image', sa.String(), nullable=True),
    sa.Column('class_name_predicted', sa.String(), nullable=True),
    sa.Column('class_name_gt', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_Image_id'), 'Image', ['id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_Image_id'), table_name='Image')
    op.drop_table('Image')
    # ### end Alembic commands ###
