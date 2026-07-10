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
    run,
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
        run=run,
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
    "research",
    "Research source facts and context before drafting.",
    pipeline.run_research,
    inputs=("transcript", "analysis"),
    outputs=("research",),
    optional=True,
    tool_name="research",
    tool_params={"project": "project id", "theme": "optional research theme"},
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
_stage("review", "Round-trip an editable script and storyboard.", None, cli_exposed=True)
_stage(
    "pair-review",
    "Review a text artifact with two distinct models.",
    pipeline.run_pair_review,
    inputs=("artifact",),
    outputs=("review",),
    optional=True,
    tool_name="pair_review",
    tool_params={
        "project": "project id",
        "artifact": "analysis|research|script|storyboard|source_draft|zack_draft",
    },
)
_stage(
    "pair-rewrite",
    "Rewrite an artifact from a pair-review report.",
    pipeline.run_pair_rewrite,
    inputs=("artifact", "review"),
    outputs=("revised-artifact",),
    optional=True,
    tool_name="pair_rewrite",
    tool_params={
        "project": "project id",
        "artifact": "research|script|source_draft|zack_draft",
        "report": "PairReviewReport",
    },
    cli_exposed=False,
)
_stage(
    "pair-verify",
    "Verify one revised artifact without triggering another rewrite.",
    pipeline.run_pair_verify,
    inputs=("revised-artifact",),
    outputs=("verification",),
    optional=True,
    tool_name="pair_verify",
    tool_params={
        "project": "project id",
        "artifact": "research|script|source_draft|zack_draft",
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
    "zack-ranking",
    "Rank radar snapshots with the configured editorial policy.",
    pipeline.run_zack_ranking,
    inputs=("snapshots", "feedback"),
    outputs=("candidates",),
    tool_name="zack_ranking",
    tool_params={"workflow": "daily-radar", "date": "YYYY-MM-DD", "top_n": "candidate count"},
)
_stage(
    "source-content",
    "Fetch subtitles or transcripts for radar candidates.",
    pipeline.run_source_content,
    inputs=("candidates", "channels"),
    outputs=("candidates", "source-content"),
    tool_name="source_content",
    tool_params={"workflow": "daily-radar", "date": "YYYY-MM-DD"},
)
_stage(
    "content-rerank",
    "Rerank the enriched metadata shortlist into final candidates.",
    pipeline.run_content_rerank,
    inputs=("shortlist", "source-content", "feedback"),
    outputs=("candidates",),
    tool_name="content_rerank",
    tool_params={"workflow": "daily-radar", "date": "YYYY-MM-DD"},
)
_stage(
    "zack-selection",
    "Choose one radar topic for research and drafting.",
    pipeline.run_zack_selection,
    inputs=("candidates",),
    outputs=("selection",),
    tool_name="zack_selection",
    tool_params={"workflow": "daily-radar", "date": "YYYY-MM-DD"},
)
_stage(
    "source-research",
    "Research the selected radar topic.",
    pipeline.run_source_research,
    inputs=("selection",),
    outputs=("source-research",),
    optional=True,
    tool_name="source_research",
    tool_params={"workflow": "daily-radar", "date": "YYYY-MM-DD"},
)
_stage(
    "zack-draft",
    "Generate the selected radar topic brief and draft.",
    pipeline.run_zack_draft,
    inputs=("candidates", "selection", "source-research"),
    outputs=("brief", "zack-draft"),
    tool_name="zack_draft",
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
    WorkflowStep("research", when="research"),
    WorkflowStep("source-draft"),
    status_name="source_draft",
)
_workflow(
    "remix",
    "Create a remixed video from one source.",
    WorkflowStep("download"),
    WorkflowStep("extract"),
    WorkflowStep("analyze"),
    WorkflowStep("research", when="research"),
    WorkflowStep("storyboard"),
    WorkflowStep("voiceover"),
    WorkflowStep("assemble"),
)
_workflow(
    "original",
    "Create a video from a theme and generated assets.",
    WorkflowStep("research", when="research"),
    WorkflowStep("storyboard"),
    WorkflowStep("assets"),
    WorkflowStep("voiceover"),
    WorkflowStep("assemble"),
)
_workflow(
    "daily-radar",
    "Collect channel signals, select one topic, and generate a draft.",
    WorkflowStep("collect"),
    WorkflowStep("zack-ranking"),
    WorkflowStep("source-content"),
    WorkflowStep("content-rerank"),
    WorkflowStep("zack-selection"),
    WorkflowStep("source-research", when="research"),
    WorkflowStep("zack-draft"),
    WorkflowStep("pair-review", when="review"),
    WorkflowStep("pair-rewrite", when="rewrite"),
    WorkflowStep("pair-verify", when="rewrite"),
)
