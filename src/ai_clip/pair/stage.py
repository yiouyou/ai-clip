from __future__ import annotations

from pathlib import Path

from ai_clip.core import billing
from ai_clip.core.config import Config
from ai_clip.core import llm as llm_mod
from ai_clip.core.artifacts import write_text_atomic
from ai_clip.core.llm import extract_json
from ai_clip.core.paths import ProjectPaths, write_model
from ai_clip.pair import client
from ai_clip.pair.models import PairReviewReport, ReviewerResult, ReviewIssue
from ai_clip.pair.prompts import LOGIC_SYSTEM, REVIEW_USER, STYLE_SYSTEM
from ai_clip.radar.storage import RadarPaths

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
    artifact = _normalize_artifact(artifact)
    pp = ProjectPaths(cfg.data_dir, project)
    pp.ensure()
    source, out = _artifact_paths(cfg, pp, artifact, run_date)
    if not source.exists():
        raise PairReviewError(f"{artifact} artifact not found: {source}")
    content = source.read_text(encoding="utf-8")
    if not content.strip():
        raise PairReviewError(f"{artifact} artifact is empty: {source}")

    models = _review_model_pool(cfg)
    if len({m.model for m in models}) < 2:
        raise PairReviewError("pair-review requires at least two distinct reviewer models")

    used: set[str] = set()
    results: list[ReviewerResult] = []
    with billing.account(pp.root, "pair_review"):
        for role, system in _ROLE_SYSTEMS.items():
            result = _run_role(
                role=role,
                system=system,
                content=content,
                artifact=artifact,
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
        artifact=artifact,
        source_path=str(source),
        producer_model=cfg.llm.model,
        status=status,
        reviewers=results,
    )
    write_model(out, report)
    return report


def rewrite_reviewed_artifact(
    cfg: Config,
    project: str,
    artifact: str,
    report: PairReviewReport,
    run_date: str | None = None,
) -> Path:
    artifact = _normalize_artifact(artifact)
    source, _ = _artifact_paths(cfg, ProjectPaths(cfg.data_dir, project), artifact, run_date)
    out = _rewrite_path(cfg, project, artifact, run_date)
    if artifact not in REWRITABLE_ARTIFACTS:
        allowed = ", ".join(sorted(REWRITABLE_ARTIFACTS))
        raise PairReviewError(f"rewrite is only supported for: {allowed}")
    if report.status == "blocked":
        raise PairReviewError("cannot rewrite artifact because pair-review is blocked")
    if not source.exists():
        raise PairReviewError(f"{artifact} artifact not found: {source}")
    content = source.read_text(encoding="utf-8")
    with billing.account(_artifact_root(cfg, project, artifact, run_date), "pair_rewrite"):
        revised = llm_mod.chat(
            cfg.llm,
            system=REWRITE_SYSTEM,
            user=REWRITE_USER.format(
                artifact=artifact,
                content=content,
                report=report.model_dump_json(indent=2),
            ),
        )
    write_text_atomic(out, revised.strip(), encoding="utf-8")
    return out


def _artifact_paths(
    cfg: Config,
    pp: ProjectPaths,
    artifact: str,
    run_date: str | None,
) -> tuple[Path, Path]:
    if artifact == "zack_draft":
        if not run_date:
            raise PairReviewError("zack_draft review requires --date")
        paths = RadarPaths(cfg.data_dir, run_date)
        return paths.existing_draft_md(), paths.reviews_dir / f"{run_date}_zack_draft_review.json"
    mapping = {
        "analysis": pp.analysis_json,
        "research": pp.research_md,
        "script": pp.script_md,
        "source_draft": pp.source_draft_md,
        "storyboard": pp.storyboard_md,
    }
    if artifact not in mapping:
        allowed_items = sorted([*mapping, "zack_draft"])
        allowed = ", ".join(allowed_items)
        raise PairReviewError(f"unknown artifact {artifact!r}; expected one of: {allowed}")
    return mapping[artifact], pp.reviews_dir / f"{artifact}_review.json"


def _rewrite_path(
    cfg: Config,
    project: str,
    artifact: str,
    run_date: str | None,
) -> Path:
    pp = ProjectPaths(cfg.data_dir, project)
    if artifact == "research":
        return pp.root / "research.revised.md"
    if artifact == "script":
        return pp.root / "script.revised.md"
    if artifact == "source_draft":
        return pp.source_draft_revised_md
    if artifact == "zack_draft":
        if not run_date:
            raise PairReviewError("zack_draft rewrite requires --date")
        return RadarPaths(cfg.data_dir, run_date).draft_revised_md
    allowed = ", ".join(sorted(REWRITABLE_ARTIFACTS))
    raise PairReviewError(f"rewrite is only supported for: {allowed}")


def _artifact_root(
    cfg: Config,
    project: str,
    artifact: str,
    run_date: str | None,
) -> Path:
    if artifact == "zack_draft":
        if not run_date:
            raise PairReviewError("zack_draft rewrite requires --date")
        return RadarPaths(cfg.data_dir, run_date).root
    return ProjectPaths(cfg.data_dir, project).root


def _normalize_artifact(artifact: str) -> str:
    if artifact in {"radar_draft", "scout_draft"}:
        return "zack_draft"
    return artifact


def _artifact_path(pp: ProjectPaths, artifact: str) -> Path:
    mapping = {
        "analysis": pp.analysis_json,
        "research": pp.research_md,
        "script": pp.script_md,
        "source_draft": pp.source_draft_md,
        "storyboard": pp.storyboard_md,
    }
    if artifact not in mapping:
        allowed = ", ".join(sorted(mapping))
        raise PairReviewError(f"unknown artifact {artifact!r}; expected one of: {allowed}")
    return mapping[artifact]


def _review_model_pool(cfg: Config) -> list[client.ReviewModel]:
    models = client.configured_models(cfg.pair)
    producer = cfg.llm.model
    without_producer = [m for m in models if m.model != producer]
    if len({m.model for m in without_producer}) >= 2:
        return without_producer
    return models


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
