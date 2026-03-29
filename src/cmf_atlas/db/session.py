"""Database engine and session management."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from cmf_atlas.db.models import Base

_DEFAULT_DB = Path(__file__).resolve().parents[3] / "data" / "atlas.db"


def get_engine(db_path: str | Path | None = None, echo: bool = False):
    """Create a SQLAlchemy engine for the given SQLite path."""
    path = Path(db_path) if db_path else _DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=echo)


def init_db(engine=None, db_path: str | Path | None = None):
    """Create all tables. Idempotent."""
    if engine is None:
        engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None, db_path: str | Path | None = None) -> Session:
    """Return a new Session bound to the given engine."""
    if engine is None:
        engine = get_engine(db_path)
    return sessionmaker(bind=engine)()
