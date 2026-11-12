from __future__ import annotations

from ai_clip import pipeline
from ai_clip.core.stages import StageRegistry, StageSpec, WorkflowSpec, WorkflowStep


REGISTRY = StageRegistry()


def stage_command(app, name: str):
    spec = REGISTRY.stage(name)
    if not spec.cli_exposed:
        raise ValueError(f"stage {name!r} is not CLI-exposed")
    return app.command(spec.name, help=spec.description)


def workflow_command(app, name: str):
    spec = REGISTRY.workflow(name)
    if not spec.cli_exposed:
        raise ValueError(f"workflow {name!r} is not CLI-exposed")
    return app.command(spec.name, help=spec.description)


def _stage(
    name: str,
    description: str,
    runner,
    *,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    optional: bool = False,
    tool_name: str | None = None,
    tool_params: dict[str, str] | None = None,
    cli_exposed: bool = True,
) -> None:
    REGISTRY.register_stage(StageSpec(
        name=name,
        description=description,
        inputs=inputs,
        outputs=outputs,
        optional=optional,
        runner=runner,
        tool_name=tool_name,
        tool_params=tool_params or {},
        cli_exposed=cli_exposed,
    ))


_stage(
    "discover",
    "Discover and rank source videos for a topic.",
    pipeline.run_discover,
    outputs=("candidates",),
    cli_exposed=True,
)
_stage(
    "download",
    "Download a source clip from a URL via yt-dlp.",
    pipeline.run_download,
    inputs=("url",),
    outputs=("clip",),
    tool_name="download",
    tool_params={"project": "project id", "url": "source video URL"},
)
_stage(
    "extract",
    "Extract or transcribe source audio.",
    pipeline.run_extract,
    inputs=("clip",),
    outputs=("transcript",),
    tool_name="extract",
    tool_params={"project": "project id"},
)
_stage("export", "Export a transcript as SRT and plain text.", pipeline.run_export)
_stage(
    "analyze",
    "Reverse-engineer the source video's content structure.",
    pipeline.run_analyze,
    inputs=("transcript",),
    outputs=("analysis",),
    tool_name="analyze",
    tool_params={"project": "project id"},
)
_stage(
    "source-research",
    "Research source facts and context before drafting.",
    pipeline.run_source_research,
    inputs=("transcript", "analysis"),
    outputs=("research",),
    optional=True,
    tool_name="source_research",
    tool_params={"project": "project id", "theme": "optional research theme"},
)
_stage(
    "topic-research",
    "Research a theme without requiring a source transcript.",
    pipeline.run_topic_research,
    inputs=("theme",),
    outputs=("research",),
    optional=True,
    cli_exposed=False,
)
_stage(
    "storyboard",
    "Generate a shot list with media prompts.",
    pipeline.run_storyboard,
    inputs=("analysis", "transcript", "research"),
    outputs=("storyboard",),
    tool_name="storyboard",
    tool_params={
        "project": "project id",
        "theme": "video theme",
        "duration_sec": "target length",
        "n_shots": "number of shots",
    },
)
_stage(
    "source-draft",
    "Generate an original talking-head draft from one source.",
    pipeline.run_source_draft,
    inputs=("transcript", "analysis", "research"),
    outputs=("source-draft",),
    tool_name="source_draft",
    tool_params={"project": "project id", "intent": "info|emotion|sales"},
    cli_exposed=False,
)
_stage(
    "assets",
    "Generate missing image assets with the configured provider.",
    pipeline.run_assets,
    inputs=("storyboard",),
    outputs=("assets",),
    tool_name="assets",
    tool_params={"project": "project id"},
)
_stage(
    "storyboard-edit",
    "Round-trip an editable script and storyboard.",
    None,
    cli_exposed=True,
)
_stage(
    "review-artifact",
    "Review a text artifact with two distinct models.",
    pipeline.run_artifact_review,
    inputs=("artifact",),
    outputs=("review",),
    optional=True,
    tool_name="review_artifact",
    tool_params={
        "project": "project id",
        "artifact": "analysis|research|script|storyboard|source-draft|editorial-draft",
    },
)
_stage(
    "rewrite-artifact",
    "Rewrite an artifact from an artifact-review report.",
    pipeline.run_artifact_rewrite,
    inputs=("artifact", "review"),
    outputs=("revised-artifact",),
    optional=True,
    tool_name="rewrite_artifact",
    tool_params={
        "project": "project id",
        "artifact": "research|script|source-draft|editorial-draft",
        "report": "ArtifactReviewReport",
    },
    cli_exposed=False,
)
_stage(
    "verify-artifact",
    "Verify one revised artifact without triggering another rewrite.",
    pipeline.run_artifact_verify,
    inputs=("revised-artifact",),
    outputs=("verification",),
    optional=True,
    tool_name="verify_artifact",
    tool_params={
        "project": "project id",
        "artifact": "research|script|source-draft|editorial-draft",
    },
    cli_exposed=False,
)
_stage(
    "voiceover",
    "Synthesize per-shot narration via configured TTS.",
    pipeline.run_voiceover,
    inputs=("storyboard",),
    outputs=("voice",),
    tool_name="voiceover",
    tool_params={"project": "project id"},
)
_stage(
    "assemble",
    "Assemble assets, source clips, voiceover, and captions into MP4.",
    pipeline.run_assemble,
    inputs=("storyboard", "assets", "voice"),
    outputs=("output",),
    tool_name="assemble",
    tool_params={"project": "project id"},
)
_stage(
    "collect",
    "Collect metadata snapshots from configured channels.",
    pipeline.run_collect,
    inputs=("channels",),
    outputs=("snapshots", "collect-report"),
    tool_name="collect",
    tool_params={"workflow": "daily-radar", "date": "YYYY-MM-DD", "force": "fetch again"},
)
_stage(
    "rank-candidates",
    "Rank radar snapshots with the configured editorial policy.",
    pipeline.run_rank_candidates,
    inputs=("snapshots", "feedback"),
    outputs=("candidates",),
    tool_name="rank_candidates",
    tool_params={"workflow": "daily-radar", "date": "YYYY-MM-DD", "top_n": "candidate count"},
)
_stage(
    "enrich-content",
    "Fetch subtitles or transcripts for radar candidates.",
    pipeline.run_enrich_content,
    inputs=("candidates", "channels"),
    outputs=("candidates", "source-content"),
    tool_name="enrich_content",
    tool_params={"workflow": "daily-radar", "date": "YYYY-MM-DD"},
)
_stage(
    "rerank-candidates",
    "Rerank the enriched metadata shortlist into final candidates.",
    pipeline.run_rerank_candidates,
    inputs=("shortlist", "source-content", "feedback"),
    outputs=("candidates",),
    tool_name="rerank_candidates",
    tool_params={"workflow": "daily-radar", "date": "YYYY-MM-DD"},
)
_stage(
    "select-topic",
    "Choose one radar topic for research and drafting.",
    pipeline.run_select_topic,
    inputs=("candidates",),
    outputs=("selection",),
    tool_name="select_topic",
    tool_params={"workflow": "daily-radar", "date": "YYYY-MM-DD"},
)
_stage(
    "research-evidence",
    "Research the selected radar topic.",
    pipeline.run_research_evidence,
    inputs=("selection",),
    outputs=("evidence-research",),
    optional=True,
    tool_name="research_evidence",
    tool_params={"workflow": "daily-radar", "date": "YYYY-MM-DD"},
)
_stage(
    "draft-script",
    "Generate the selected radar topic brief and draft.",
    pipeline.run_draft_script,
    inputs=("candidates", "selection", "evidence-research"),
    outputs=("brief", "editorial-draft"),
    tool_name="draft_script",
    tool_params={"workflow": "daily-radar", "date": "YYYY-MM-DD"},
)


def _workflow(
    name: str,
    description: str,
    *steps: WorkflowStep,
    status_name: str = "",
) -> None:
    REGISTRY.register_workflow(WorkflowSpec(
        name=name,
        description=description,
        steps=steps,
        status_name=status_name,
    ))


_workflow(
    "transcribe",
    "Download, transcribe, and export subtitles and text.",
    WorkflowStep("download"),
    WorkflowStep("extract"),
    WorkflowStep("export"),
)
_workflow(
    "teardown",
    "Download, transcribe, and analyze a source video.",
    WorkflowStep("download"),
    WorkflowStep("extract"),
    WorkflowStep("analyze"),
)
_workflow(
    "source-draft",
    "Turn one source video into an original talking-head draft.",
    WorkflowStep("download"),
    WorkflowStep("extract"),
    WorkflowStep("analyze"),
    WorkflowStep("source-research", when="research"),
    WorkflowStep("source-draft"),
    status_name="source-draft",
)
_workflow(
    "remix",
    "Create a remixed video from one source.",
    WorkflowStep("download"),
    WorkflowStep("extract"),
    WorkflowStep("analyze"),
    WorkflowStep("source-research", when="research"),
    WorkflowStep("storyboard"),
    WorkflowStep("voiceover"),
    WorkflowStep("assemble"),
)
_workflow(
    "original",
    "Create a video from a theme and generated assets.",
    WorkflowStep("topic-research", when="research"),
    WorkflowStep("storyboard"),
    WorkflowStep("assets"),
    WorkflowStep("voiceover", when="assets-ready"),
    WorkflowStep("assemble", when="assets-ready"),
)
_workflow(
    "daily-radar",
    "Collect channel signals, select one topic, and generate a draft.",
    WorkflowStep("collect"),
    WorkflowStep("rank-candidates"),
    WorkflowStep("enrich-content"),
    WorkflowStep("rerank-candidates"),
    WorkflowStep("select-topic"),
    WorkflowStep("research-evidence", when="research"),
    WorkflowStep("draft-script"),
    WorkflowStep("review-artifact", when="review"),
    WorkflowStep("rewrite-artifact", when="rewrite"),
    WorkflowStep("verify-artifact", when="rewrite"),
)
