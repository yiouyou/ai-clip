from __future__ import annotations

from pathlib import Path

from ai_clip.core import billing
from ai_clip.core.config import Config
from ai_clip.core import llm as llm_mod
from ai_clip.core.artifacts import (
    ArtifactRef,
    artifact_matches,
    write_artifact_manifest,
    write_text_atomic,
)
from ai_clip.core.llm import extract_json
from ai_clip.core.paths import write_model
from ai_clip.pair import client
from ai_clip.pair.models import PairReviewReport, ReviewerResult, ReviewIssue
from ai_clip.pair.prompts import LOGIC_SYSTEM, REVIEW_USER, STYLE_SYSTEM

_ROLE_SYSTEMS = {
    "logic": LOGIC_SYSTEM,
    "style": STYLE_SYSTEM,
}

REWRITE_SYSTEM = (
    "You revise short-video production artifacts. Preserve the creator's core angle "
    "and intent, but apply the reviewers' concrete fixes for accuracy, structure, "
    "hook, pacing, audience fit, and safe wording. Return only the revised Markdown."
)

REWRITE_USER = """Revise this {artifact} using the review report.

Original draft:
```markdown
{content}
```

Review report JSON:
```json
{report}
```
"""


class PairReviewError(RuntimeError):
    pass


REWRITABLE_ARTIFACTS = {"research", "script", "source_draft", "zack_draft"}


def review_artifact(
    cfg: Config,
    project: str,
    artifact: str,
    run_date: str | None = None,
) -> PairReviewReport:
    target = _resolve_artifact(cfg, project, artifact, run_date)
    return review_artifact_ref(cfg, project, target)


def review_artifact_ref(
    cfg: Config,
    project: str,
    target: ArtifactRef,
) -> PairReviewReport:
    return _review_path(
        cfg,
        project,
        target,
        source=target.source,
        output=target.review,
        kind="review",
        billing_stage="pair_review",
    )


def verify_rewritten_artifact(
    cfg: Config,
    project: str,
    artifact: str,
    run_date: str | None = None,
) -> PairReviewReport:
    target = _resolve_artifact(cfg, project, artifact, run_date)
    if target.revised is None or target.verification is None:
        raise PairReviewError(f"{target.name} does not support rewrite verification")
    return _review_path(
        cfg,
        project,
        target,
        source=target.revised,
        output=target.verification,
        kind="verify",
        billing_stage="pair_verify",
    )


def _review_path(
    cfg: Config,
    project: str,
    target: ArtifactRef,
    *,
    source: Path,
    output: Path,
    kind: str,
    billing_stage: str,
) -> PairReviewReport:
    if not source.exists():
        raise PairReviewError(f"{target.name} artifact not found: {source}")
    content = source.read_text(encoding="utf-8")
    if not content.strip():
        raise PairReviewError(f"{target.name} artifact is empty: {source}")

    models = _review_model_pool(cfg)
    if len({m.model for m in models}) < 2:
        raise PairReviewError("pair-review requires at least two distinct reviewer models")

    used: set[str] = set()
    results: list[ReviewerResult] = []
    with billing.account(target.billing_root, billing_stage):
        for role, system in _ROLE_SYSTEMS.items():
            result = _run_role(
                role=role,
                system=system,
                content=content,
                artifact=target.name,
                project=project,
                producer_model=cfg.llm.model,
                models=models,
                used_models=used,
                timeout=cfg.pair.timeout,
            )
            if result.model:
                used.add(result.model)
            results.append(result)

    status = "passed" if results and all(r.ok and r.verdict == "pass" for r in results) else "needs_review"
    if any(r.ok and r.verdict == "block" for r in results) or any(not r.ok for r in results):
        status = "blocked"

    report = PairReviewReport(
        artifact=target.name,
        source_path=str(source),
        kind=kind,
        producer_model=cfg.llm.model,
        status=status,
        reviewers=results,
    )
    write_model(output, report)
    write_artifact_manifest(
        output,
        stage=f"pair-{kind}",
        inputs=[source],
        params=_review_params(cfg, kind),
        model=",".join(result.model for result in results if result.model),
    )
    return report


def rewrite_reviewed_artifact(
    cfg: Config,
    project: str,
    artifact: str,
    report: PairReviewReport,
    run_date: str | None = None,
) -> Path:
    target = _resolve_artifact(cfg, project, artifact, run_date)
    return rewrite_artifact_ref(cfg, target, report)


def rewrite_artifact_ref(
    cfg: Config,
    target: ArtifactRef,
    report: PairReviewReport,
) -> Path:
    if target.name not in REWRITABLE_ARTIFACTS or target.revised is None:
        allowed = ", ".join(sorted(REWRITABLE_ARTIFACTS))
        raise PairReviewError(f"rewrite is only supported for: {allowed}")
    if report.status == "blocked":
        raise PairReviewError("cannot rewrite artifact because pair-review is blocked")
    if not target.source.exists():
        raise PairReviewError(f"{target.name} artifact not found: {target.source}")
    content = target.source.read_text(encoding="utf-8")
    with billing.account(target.billing_root, "pair_rewrite"):
        revised = llm_mod.chat(
            cfg.llm,
            system=REWRITE_SYSTEM,
            user=REWRITE_USER.format(
                artifact=target.name,
                content=content,
                report=report.model_dump_json(indent=2),
            ),
        )
    write_text_atomic(target.revised, revised.strip(), encoding="utf-8")
    write_artifact_manifest(
        target.revised,
        stage="pair-rewrite",
        inputs=[target.source, target.review],
        params={"artifact": target.name, "prompt_version": "1"},
        model=cfg.llm.model,
    )
    return target.revised


def pair_artifact_freshness(cfg: Config, target: ArtifactRef) -> dict[str, bool]:
    review = artifact_matches(
        target.review,
        inputs=[target.source],
        params=_review_params(cfg, "review"),
    )
    rewrite = bool(
        target.revised
        and artifact_matches(
            target.revised,
            inputs=[target.source, target.review],
            params={"artifact": target.name, "prompt_version": "1"},
            model=cfg.llm.model,
        )
    )
    verify = bool(
        rewrite
        and target.revised
        and target.verification
        and artifact_matches(
            target.verification,
            inputs=[target.revised],
            params=_review_params(cfg, "verify"),
        )
    )
    return {"review": review, "rewrite": rewrite, "verify": verify}


def _resolve_artifact(
    cfg: Config,
    project: str,
    artifact: str,
    run_date: str | None,
) -> ArtifactRef:
    from ai_clip.artifact_catalog import resolve_review_artifact

    try:
        return resolve_review_artifact(cfg, project, artifact, run_date)
    except ValueError as exc:
        raise PairReviewError(str(exc)) from exc


def _review_model_pool(cfg: Config) -> list[client.ReviewModel]:
    models = client.configured_models(cfg.pair)
    producer = cfg.llm.model
    without_producer = [m for m in models if m.model != producer]
    if len({m.model for m in without_producer}) >= 2:
        return without_producer
    return models


def _review_params(cfg: Config, kind: str) -> dict[str, str]:
    models = client.configured_models(cfg.pair)
    return {
        "kind": kind,
        "producer_model": cfg.llm.model,
        "reviewer_pool": ",".join(f"{model.base_url.rstrip('/')}|{model.model}" for model in models),
    }


def _run_role(
    role: str,
    system: str,
    content: str,
    artifact: str,
    project: str,
    producer_model: str,
    models: list[client.ReviewModel],
    used_models: set[str],
    timeout: float,
) -> ReviewerResult:
    errors: list[str] = []
    for model in models:
        if model.model in used_models:
            continue
        try:
            raw = client.chat(
                model,
                system=system,
                user=REVIEW_USER.format(
                    artifact=artifact,
                    project=project,
                    producer_model=producer_model,
                    content=content,
                ),
                timeout=timeout,
            )
            return _parse_result(role, model.model, raw)
        except Exception as exc:
            errors.append(f"{model.model}: {exc}")
    return ReviewerResult(role=role, ok=False, error="; ".join(errors))


def _parse_result(role: str, model: str, raw: str) -> ReviewerResult:
    data = extract_json(raw)
    verdict = str(data.get("verdict", "revise")).strip().lower()
    if verdict not in {"pass", "revise", "block"}:
        verdict = "revise"
    issues = [
        ReviewIssue(
            severity=str(item.get("severity", "medium")),
            category=str(item.get("category", "")),
            detail=str(item.get("detail", "")),
            suggestion=str(item.get("suggestion", "")),
        )
        for item in data.get("issues", [])
        if isinstance(item, dict)
    ]
    return ReviewerResult(
        role=role,
        model=model,
        ok=True,
        verdict=verdict,
        summary=str(data.get("summary", "")),
        issues=issues,
        raw=raw,
    )
