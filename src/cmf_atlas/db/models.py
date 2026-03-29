"""SQLAlchemy ORM models — matches the DDL spec exactly."""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "project"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    series = relationship("Series", back_populates="project", cascade="all, delete-orphan")


class Series(Base):
    __tablename__ = "series"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("project.id"), nullable=False)
    name = Column(Text)
    definition = Column(Text)
    generator_type = Column(Text, nullable=False)
    provenance = Column(Text)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    project = relationship("Project", back_populates="series")
    representations = relationship("Representation", back_populates="series", cascade="all, delete-orphan")


class Representation(Base):
    __tablename__ = "representation"

    id = Column(Integer, primary_key=True)
    series_id = Column(Integer, ForeignKey("series.id"), nullable=False)
    primary_group = Column(Text, nullable=False)  # "dfinite" | "hypergeometric" | "pcf"
    canonical_fingerprint = Column(Text, nullable=False)
    canonical_payload = Column(Text, nullable=False)  # JSON
    overlap_groups = Column(Text)  # JSON array
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("primary_group", "canonical_fingerprint", name="uq_repr_fingerprint"),
    )

    series = relationship("Series", back_populates="representations")
    features = relationship("Features", back_populates="representation", uselist=False, cascade="all, delete-orphan")
    cmfs = relationship("CMF", back_populates="representation", cascade="all, delete-orphan")
    equivalence_links = relationship("RepresentationEquivalence", back_populates="representation", cascade="all, delete-orphan")


class Features(Base):
    __tablename__ = "features"

    representation_id = Column(Integer, ForeignKey("representation.id"), primary_key=True)
    feature_json = Column(Text, nullable=False)  # JSON map
    feature_version = Column(Text, nullable=False)
    computed_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    representation = relationship("Representation", back_populates="features")


class CMF(Base):
    __tablename__ = "cmf"

    id = Column(Integer, primary_key=True)
    representation_id = Column(Integer, ForeignKey("representation.id"), nullable=False)
    cmf_payload = Column(Text, nullable=False)  # JSON
    dimension = Column(Integer)
    direction_policy = Column(Text)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    representation = relationship("Representation", back_populates="cmfs")
    eval_runs = relationship("EvalRun", back_populates="cmf", cascade="all, delete-orphan")


class EvalRun(Base):
    __tablename__ = "eval_run"

    id = Column(Integer, primary_key=True)
    cmf_id = Column(Integer, ForeignKey("cmf.id"), nullable=False)
    run_type = Column(Text, nullable=False)  # "quick", "standard", "high_precision"
    precision_digits = Column(Integer, nullable=False)
    steps = Column(Integer)
    limit_estimate = Column(Text)  # string repr high-precision
    error_estimate = Column(Float)
    convergence_score = Column(Float)  # 0..1
    stability_score = Column(Float)  # 0..1
    runtime_ms = Column(Integer)
    notes = Column(Text)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    cmf = relationship("CMF", back_populates="eval_runs")
    recognition_attempts = relationship("RecognitionAttempt", back_populates="eval_run", cascade="all, delete-orphan")


class RecognitionAttempt(Base):
    __tablename__ = "recognition_attempt"

    id = Column(Integer, primary_key=True)
    eval_run_id = Column(Integer, ForeignKey("eval_run.id"), nullable=False)
    method = Column(Text, nullable=False)  # "pslq", "isc_lookup", "rule_transform"
    basis_name = Column(Text)
    basis_payload = Column(Text)  # JSON
    success = Column(Integer, nullable=False)  # 0/1
    identified_as = Column(Text)
    relation_height = Column(Float)
    residual_log10 = Column(Float)
    attempt_log = Column(Text)  # verbose audit
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    eval_run = relationship("EvalRun", back_populates="recognition_attempts")


class EquivalenceClass(Base):
    __tablename__ = "equivalence_class"

    id = Column(Integer, primary_key=True)
    primary_group = Column(Text, nullable=False)
    class_fingerprint = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("primary_group", "class_fingerprint", name="uq_equiv_class"),
    )

    members = relationship("RepresentationEquivalence", back_populates="equivalence_class", cascade="all, delete-orphan")


class RepresentationEquivalence(Base):
    __tablename__ = "representation_equivalence"

    representation_id = Column(Integer, ForeignKey("representation.id"), primary_key=True)
    equivalence_class_id = Column(Integer, ForeignKey("equivalence_class.id"), primary_key=True)

    representation = relationship("Representation", back_populates="equivalence_links")
    equivalence_class = relationship("EquivalenceClass", back_populates="members")
