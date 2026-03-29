"""Database layer — SQLAlchemy ORM models and session management."""

from cmf_atlas.db.models import (
    Base,
    Project,
    Series,
    Representation,
    Features,
    CMF,
    EvalRun,
    RecognitionAttempt,
    EquivalenceClass,
    RepresentationEquivalence,
)
from cmf_atlas.db.session import get_engine, get_session, init_db

__all__ = [
    "Base",
    "Project",
    "Series",
    "Representation",
    "Features",
    "CMF",
    "EvalRun",
    "RecognitionAttempt",
    "EquivalenceClass",
    "RepresentationEquivalence",
    "get_engine",
    "get_session",
    "init_db",
]
