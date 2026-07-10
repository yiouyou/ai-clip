# ai-clip

**English** | [中文](README.md)

ai-clip is a lightweight short-video pipeline orchestrator. It can remix viral clips
or produce original videos from a topic. The project owns contracts, CLI commands,
device adaptation, prompts, cost logging, and workflow status; heavy work is delegated
to yt-dlp, faster-whisper, ComfyUI, ffmpeg, LLM APIs, and TTS APIs.

```text
discover -> download -> extract -> analyze -> research -> storyboard -> review -> assets -> voiceover -> assemble
```

Detailed docs:

- [Workflows](docs/workflows.md)
- [Research](docs/research.md)
- [Daily Radar](docs/radar.md)
- [Pair Review](docs/pair-review.md)

## Principles

- **Prompt-first, human-in-the-loop**: `storyboard.md`, `script.md`, and `research.md` are editable.
- **Stable filename contract**: shot N uses `assets/shot_NN.png` / `shot_NN.mp4`.
- **Per-project artifacts**: everything lives under `data/<project>/`.
- **Lightweight metadata**: key artifacts write `<artifact>.meta.json` for freshness checks.
- **Observable workflows**: composed workflows write `data/<project>/runs/<workflow>.json`.

## Requirements

- Python 3.12+ and [uv](https://github.com/astral-sh/uv)
- **ffmpeg + ffprobe** on PATH
- Optional: OpenAI-compatible LLM key, Tavily key, MiMo TTS key, local ComfyUI

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv pip install -e ".[download,extract,llm]"
cp .env.example .env
```

## Quick Start

Remix:

```bash
ai-clip remix "<clip-url>" -p demo --theme "explain compounding in 60s"
ai-clip status -p demo
ai-clip assemble -p demo
```

Original:

```bash
ai-clip original -p promo --theme "city night-ride vlog opener" --shots 5
```

Single-source original draft:

```bash
ai-clip source-draft "<clip-url>" -p demo --research --theme "explain this through complex systems"
ai-clip source-draft "<clip-url>" -p demo --whisper-model small   # faster CPU transcription for long videos
```

Project research:

```bash
ai-clip research -p demo --theme "AI companies as ecological niches" --max-searches 2
ai-clip storyboard -p demo --theme "AI companies as ecological niches"
```

Daily topic radar:

```bash
ai-clip daily-radar --top 3 --research --research-searches 1
```

## Common Commands

| Command | Purpose |
|---------|---------|
| `ai-clip transcribe <url> -p P` | Download, transcribe, and export `.srt` / `.txt` |
| `ai-clip teardown <url> -p P` | Download, transcribe, and analyze viral formula |
| `ai-clip source-draft <url> -p P` | Generate an original talking-head draft from one source; reuses existing artifacts by default, use `--no-resume` to force a rerun |
| `ai-clip research -p P --theme T` | Write `research.json` and editable `research.md` |
| `ai-clip storyboard -p P --theme T` | Generate storyboard, prompts, and `storyboard.md` |
| `ai-clip review -p P` / `--apply` | Round-trip `storyboard.json` through `script.md` |
| `ai-clip pair-review -p P --artifact script` | Multi-model review of text artifacts |
| `ai-clip status -p P` | Show artifact freshness and missing assets |
| `ai-clip status -p P --json` | Emit machine-readable project status |
| `ai-clip assets -p P` | Generate missing image assets |
| `ai-clip voiceover -p P` | Generate voiceover |
| `ai-clip assemble -p P` | Assemble final mp4 |
| `ai-clip doctor` | Diagnose local environment |

## Video Formats

`storyboard`, `remix`, and `original` support `--format`:

| Format | Output | Assets |
|--------|--------|--------|
| `talking_head` | narration lines + optional b-roll | optional stills |
| `slideshow` | image cards + captions + narration | one image per card |
| `remix` | source spans + fresh narration | none; uses source video |
| `montage` | fully generated multi-shot drama | image/video per shot |

## Configuration

Config loading order:

1. `config/default.yaml`
2. `.env`
3. CLI arguments

Common environment variables:

```text
AICLIP_LLM_BASE_URL
AICLIP_LLM_MODEL
AICLIP_DATA_DIR
DEEPSEEK_API_KEY
OPENAI_API_KEY
NEWAPP_URL
NEWAPP_API_KEY
TAVILY_API_KEY
MIMO_API_KEY
AICLIP_COMFYUI_URL
```

`NEWAPP_URL` should be the full OpenAI-compatible base URL, including `/v1`.

## Local Checks

Windows smoke test with no paid API calls:

```powershell
.\scripts\smoke.ps1
```

Before submitting changes:

```bash
uv run ruff check src tests
uv run --extra download pytest -q
```

## Optional Backends

- **ComfyUI**: local image generation via `assets.image_provider: comfyui` or `auto`.
- **Smart Illustrator**: infographic/thumbnail image provider through `assets.smart_illustrator_*`.
- **MoneyPrinterTurbo / NarratoAI**: external produce backends for comparison.

## Cost Accounting

LLM/TTS calls append to `data/<project>/cost.jsonl`:

```bash
ai-clip cost -p demo
```

## License & Dependencies

ai-clip orchestrates third-party projects such as yt-dlp, faster-whisper, ComfyUI,
and ffmpeg. Review each upstream license before redistribution or commercial use.
