"""Simple migration helper — creates/upgrades tables."""

from cmf_atlas.db.session import get_engine, init_db


def migrate(db_path=None):
    """Ensure all tables exist. Safe to call repeatedly."""
    engine = get_engine(db_path)
    init_db(engine)
    print(f"Database ready: {engine.url}")
    return engine


if __name__ == "__main__":
    migrate()
