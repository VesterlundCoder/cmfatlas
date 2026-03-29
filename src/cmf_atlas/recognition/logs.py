"""Recognition audit logging — store every attempt for publishable evidence."""

from cmf_atlas.db.models import EvalRun, RecognitionAttempt
from cmf_atlas.util.json import dumps


def store_recognition_attempt(
    session,
    eval_run_id: int,
    result: dict,
) -> RecognitionAttempt:
    """Store a recognition attempt result in the database.

    Parameters
    ----------
    session : SQLAlchemy session
    eval_run_id : ID of the associated eval run
    result : dict from pslq.run_pslq() or similar

    Returns
    -------
    RecognitionAttempt ORM object
    """
    attempt = RecognitionAttempt(
        eval_run_id=eval_run_id,
        method=result.get("method", "pslq"),
        basis_name=result.get("basis_name", "standard_v1"),
        basis_payload=dumps(result.get("relation", {})),
        success=int(result.get("success", 0)),
        identified_as=result.get("identified_as"),
        relation_height=result.get("relation_height"),
        residual_log10=result.get("residual_log10"),
        attempt_log=result.get("attempt_log", ""),
    )
    session.add(attempt)
    return attempt


def store_eval_run(
    session,
    cmf_id: int,
    eval_result: dict,
    run_type: str = "quick",
) -> EvalRun:
    """Store an evaluation run result in the database."""
    run = EvalRun(
        cmf_id=cmf_id,
        run_type=run_type,
        precision_digits=eval_result.get("dps", 50),
        steps=eval_result.get("depth"),
        limit_estimate=eval_result.get("limit_estimate"),
        error_estimate=eval_result.get("error_estimate"),
        convergence_score=eval_result.get("convergence_score"),
        stability_score=eval_result.get("stability_score"),
        runtime_ms=eval_result.get("runtime_ms"),
    )
    session.add(run)
    return run
