from alembic import context

from rapmat.storage.models import Base

config = context.config
target_metadata = Base.metadata


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return

    from sqlalchemy import create_engine

    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError(
            "No connection provided and no sqlalchemy.url configured. "
            "Set it in alembic.ini for CLI use."
        )
    engine = create_engine(url)
    with engine.begin() as conn:
        _run(conn)


run_migrations_online()
