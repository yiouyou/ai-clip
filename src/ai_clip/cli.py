"""ai-clip command line. Each stage is a subcommand; `remix` and `original`
chain them. Human-in-the-loop is explicit: `storyboard` pauses for asset
creation, `assemble` resumes once assets/ is filled."""

from __future__ import annotations

import typer
from rich.console import Console

from ai_clip.core.config import load_config, load_product
from ai_clip.produce.assemble import MissingAssetsError, check_assets
from ai_clip.core.models import Intent, Platform, Storyboard, VideoFormat
from ai_clip.core.paths import ProjectPaths, read_model
from ai_clip import pipeline, workflows

app = typer.Typer(add_completion=False, help="Download, analyze and produce short videos.")
console = Console()


def _cfg(config: str | None):
    return load_config(config)


@app.command()
def discover(
    topic: str,
    project: str = typer.Option(..., "--project", "-p"),
    platform: Platform = typer.Option(Platform.youtube, "--platform"),
    channel: str = typer.Option(None, "--channel"),
    since_days: int = typer.Option(7, "--since-days"),
    limit: int = typer.Option(15, "--limit"),
    top_n: int = typer.Option(5, "--top"),
    config: str = typer.Option(None, "--config"),
):
    """Search a platform for recently-spreading clips on a topic, ranked by virality."""
    result = pipeline.run_discover(
        _cfg(config), project, topic, platform, channel, since_days, limit, top_n
    )
    if not result.candidates:
        console.print("[yellow]no candidates found[/] (try widening --since-days)")
        return
    for i, c in enumerate(result.candidates, start=1):
        console.print(
            f"[green]{i}.[/] v={c.virality:,.0f} · {c.view_count:,} views · "
            f"{c.age_days:.1f}d · {c.title[:60]}\n   {c.url}"
        )


@app.command()
def download(
    url: str,
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(None, "--config"),
):
    """Download a source clip with yt-dlp."""
    clip = pipeline.run_download(_cfg(config), project, url)
    console.print(f"[green]downloaded[/] {clip.platform} -> {clip.video_path}")


@app.command()
def extract(
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(None, "--config"),
):
    """Split audio and transcribe with faster-whisper."""
    t = pipeline.run_extract(_cfg(config), project)
    console.print(f"[green]transcribed[/] {len(t.segments)} segments, lang={t.language}")


@app.command()
def export(
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(None, "--config"),
):
    """Export the transcript to .srt and .txt."""
    srt, txt = pipeline.run_export(_cfg(config), project)
    console.print(f"[green]exported[/] {srt} , {txt}")


@app.command()
def analyze(
    project: str = typer.Option(..., "--project", "-p"),
    intent: Intent = typer.Option(Intent.info, "--intent", "-i"),
    config: str = typer.Option(None, "--config"),
):
    """Reverse-engineer the viral formula via LLM (intent: info|emotion|sales)."""
    a = pipeline.run_analyze(_cfg(config), project, intent)
    console.print(f"[green]analyzed[/] ({a.intent}) hook: {a.hook[:80]}")


@app.command()
def storyboard(
    project: str = typer.Option(..., "--project", "-p"),
    theme: str = typer.Option(..., "--theme"),
    fmt: VideoFormat = typer.Option(VideoFormat.talking_head, "--format", "-f"),
    intent: Intent = typer.Option(Intent.info, "--intent", "-i"),
    stance: str = typer.Option("", "--stance"),
    product: str = typer.Option(None, "--product"),
    duration: float = typer.Option(30.0, "--duration"),
    shots: int = typer.Option(6, "--shots"),
    config: str = typer.Option(None, "--config"),
):
    """Generate a storyboard for the chosen format (talking_head|slideshow|remix|montage)."""
    cfg = _cfg(config)
    sb = pipeline.run_storyboard(
        cfg, project, theme, fmt=fmt, intent=intent, stance=stance,
        product=load_product(product), duration_sec=duration, n_shots=shots,
    )
    pp = ProjectPaths(cfg.data_dir, project)
    console.print(f"[green]storyboard[/] ({sb.format}) {len(sb.shots)} shots -> {pp.storyboard_md}")
    if sb.format == VideoFormat.remix:
        console.print("[yellow]Next:[/] `ai-clip voiceover` then `ai-clip assemble` (no manual assets).")
    else:
        console.print(
            "[yellow]Next:[/] create assets (即梦/Gemini/ComfyUI) into "
            f"{pp.assets_dir} per storyboard.md, then `ai-clip voiceover` + `ai-clip assemble`."
        )


@app.command()
def status(
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(None, "--config"),
):
    """Show which shot assets are still missing."""
    pp = ProjectPaths(_cfg(config).data_dir, project)
    sb = read_model(pp.storyboard_json, Storyboard)
    missing = check_assets(sb, pp.assets_dir)
    if missing:
        console.print(f"[yellow]missing {len(missing)}/{len(sb.shots)}:[/]")
        for m in missing:
            console.print(f"  - {m}")
    else:
        console.print(f"[green]all {len(sb.shots)} shots have assets — ready to assemble[/]")


@app.command()
def voiceover(
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(None, "--config"),
):
    """Synthesize per-shot narration via MiMo TTS (clones the source voice by default)."""
    produced = pipeline.run_voiceover(_cfg(config), project)
    console.print(f"[green]voiceover[/] synthesized {len(produced)} shot(s)")


@app.command()
def assemble(
    project: str = typer.Option(..., "--project", "-p"),
    captions: bool = typer.Option(False, "--captions"),
    config: str = typer.Option(None, "--config"),
):
    """Stitch collected assets into the final MP4."""
    cfg = _cfg(config)
    cfg.burn_captions = captions or cfg.burn_captions
    try:
        out = pipeline.run_assemble(cfg, project)
    except MissingAssetsError as exc:
        console.print(f"[red]cannot assemble[/] — {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]assembled[/] -> {out}")


# ---- composed workflows (W1-W5) ----

@app.command()
def transcribe(
    url: str,
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(None, "--config"),
):
    """W1 提文案: download -> extract -> export srt/txt."""
    r = workflows.transcribe(_cfg(config), project, url)
    console.print(f"[green]transcribed[/] -> {r['srt']} , {r['txt']}")


@app.command()
def teardown(
    url: str,
    project: str = typer.Option(..., "--project", "-p"),
    intent: Intent = typer.Option(Intent.info, "--intent", "-i"),
    config: str = typer.Option(None, "--config"),
):
    """W2 爆款拆解: download -> extract -> analyze (intent: info|emotion|sales)."""
    r = workflows.teardown(_cfg(config), project, url, intent)
    console.print(f"[green]teardown[/] hook: {r['hook'][:80]}")
    console.print(f"formula: {r['formula'][:120]}")


@app.command()
def remix(
    url: str,
    theme: str = typer.Option(..., "--theme"),
    project: str = typer.Option(..., "--project", "-p"),
    intent: Intent = typer.Option(Intent.info, "--intent", "-i"),
    stance: str = typer.Option("", "--stance"),
    product: str = typer.Option(None, "--product"),
    captions: bool = typer.Option(False, "--captions"),
    duration: float = typer.Option(30.0, "--duration"),
    shots: int = typer.Option(6, "--shots"),
    config: str = typer.Option(None, "--config"),
):
    """W3 二创(全自动): download -> ... -> remix storyboard -> cloned voiceover -> mp4."""
    cfg = _cfg(config)
    cfg.burn_captions = captions or cfg.burn_captions
    r = workflows.remix(
        cfg, project, url, theme, intent=intent, stance=stance,
        product=load_product(product), duration=duration, n_shots=shots,
    )
    console.print(f"[green]remix done[/] -> {r['output']}")


@app.command()
def original(
    theme: str = typer.Option(..., "--theme"),
    project: str = typer.Option(..., "--project", "-p"),
    fmt: VideoFormat = typer.Option(VideoFormat.talking_head, "--format", "-f"),
    intent: Intent = typer.Option(Intent.info, "--intent", "-i"),
    stance: str = typer.Option("", "--stance"),
    product: str = typer.Option(None, "--product"),
    captions: bool = typer.Option(False, "--captions"),
    duration: float = typer.Option(30.0, "--duration"),
    shots: int = typer.Option(6, "--shots"),
    config: str = typer.Option(None, "--config"),
):
    """W4/W5 原创: storyboard -> assets(ComfyUI if available) -> voiceover -> assemble."""
    if fmt == VideoFormat.remix:
        console.print("[red]remix format needs a source clip; use `ai-clip remix`.[/]")
        raise typer.Exit(1)
    cfg = _cfg(config)
    cfg.burn_captions = captions or cfg.burn_captions
    r = workflows.original(
        cfg, project, theme, fmt=fmt, intent=intent, stance=stance,
        product=load_product(product), duration=duration, n_shots=shots,
    )
    if r["status"] == "done":
        console.print(f"[green]original done[/] -> {r['output']}")
    else:
        console.print(
            f"[yellow]need assets[/] ({len(r['missing'])} missing, {r['generated']} generated). "
            f"Fill {r['assets_dir']} per storyboard.md, then `ai-clip assemble`."
        )


@app.command()
def mpt(
    theme: str = typer.Option(..., "--theme"),
    project: str = typer.Option(..., "--project", "-p"),
    voice: str = typer.Option("", "--voice"),
    config: str = typer.Option(None, "--config"),
):
    """External backend: generate a theme video via MoneyPrinterTurbo's API."""
    from ai_clip.produce.backends import MoneyPrinterBackend, ProduceSpec

    cfg = _cfg(config)
    pp = ProjectPaths(cfg.data_dir, project)
    pp.ensure()
    backend = MoneyPrinterBackend(cfg.produce.moneyprinter_url)
    if not MoneyPrinterBackend.is_available(cfg.produce.moneyprinter_url):
        console.print(
            f"[red]MoneyPrinterTurbo not reachable[/] at {cfg.produce.moneyprinter_url}. "
            "Start it (docker compose up in the MoneyPrinterTurbo repo)."
        )
        raise typer.Exit(1)
    out = backend.produce(ProduceSpec(
        theme=theme, out_path=pp.output_mp4, aspect_ratio=cfg.aspect_ratio, voice_name=voice,
    ))
    console.print(f"[green]mpt done[/] -> {out}")


if __name__ == "__main__":
    app()
