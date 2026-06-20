# AGENTS.md — ai-clip handoff brief

Short-video pipeline orchestrator. Chains open-source tools to **remix** a viral
clip (二创) or produce **original** videos from a theme. Python 3.12, managed with
`uv`. Repo: https://github.com/yiouyou/ai-clip (branch `main`).

This file briefs an automated coding agent taking over the work. Read it fully
before changing code.

## What it is (one paragraph)

ai-clip is a **lightweight orchestrator**: it owns a data contract, a Typer CLI,
device adaptation, prompts, billing — and delegates heavy work to mature projects
(yt-dlp, faster-whisper, ffmpeg, ComfyUI) and to LLM/TTS HTTP APIs. The default
flow is **prompt-first, human-in-the-loop**: stages write human-editable files
(`storyboard.md`, `script.md`, prompt txts) and the human can intervene before the
irreversible render. It is NOT a monolithic app — keep it that way.

## Pipeline & file map

```
discover → download → extract → analyze → storyboard → [review] → (assets) → voiceover → assemble
```

| Stage | Module | Tool used | Notes |
|-------|--------|-----------|-------|
| discover | `discover/` (youtube, bilibili, social=douyin/kuaishou, ranking) | yt-dlp | keyword search (YT/B站); douyin/kuaishou = channel-listing only, often unsupported → degrades to NotImplementedError |
| download | `download/downloader.py` | yt-dlp | → `Clip` |
| extract | `extract/extractor.py` (+ `subtitles.py`, `export.py`) | ffmpeg + faster-whisper | `--subs` reuses the video's own subtitles instead of whisper |
| analyze | `analyze/` (analyzer + prompts) | LLM | viral teardown → `ViralAnalysis`; `--intent info\|emotion\|sales` |
| storyboard | `produce/storyboard.py` + `produce/formats/*` | LLM | format-aware → `Storyboard` of `Shot`s |
| review | `produce/review.py` | — | round-trips storyboard ↔ editable `script.md` (文案 human edit) |
| assets | `produce/assets/*` (factory, comfyui, prompt_only) | ComfyUI / human | images for talking_head/slideshow/montage |
| voiceover | `produce/voiceover.py` + `produce/tts/mimo.py` | MiMo TTS | clones the source speaker by default |
| assemble | `produce/assemble.py` (+ `captions.py`, `core/fonts.py`) | ffmpeg | concat, audio, optional burned captions |

Orchestration: `pipeline.py` (`run_*` fns, each opens a `billing.account()` block).
Composed workflows: `workflows.py` (transcribe/teardown/remix/original).
Tool registry for a future agent layer: `tools.py`.
External produce backends (optional, compared to self-built): `produce/backends/`
(`moneyprinter.py` REST adapter, `narrato.py` subprocess hard-wire).

### Video formats (`--format`)
`talking_head` (default, narration + optional b-roll still), `slideshow` (image
cards + caption + narration), `remix` (cut spans from the SOURCE clip + new
narration; no asset generation), `montage` (fully AI-generated). Prompts for all
four live in `produce/formats/prompts.py`; format modules keep only the
LLM-output→Shot mapping.

## Hard conventions (do not break)

- **Filename contract**: shot N ⇒ `assets/shot_NN.png` / `shot_NN.mp4`; remix shots
  carry `source_start/source_end` and need no assets. `assemble.check_assets` keys
  on `Shot.expected_files()`.
- **Per-project artifacts** under `data/<project>/`: `clip.json`, `transcript.json`,
  `analysis.json`, `storyboard.json` (+ `.md`), `script.md`, `prompts/`, `assets/`,
  `voice/`, `candidates.json`, `cost.jsonl`, `output.mp4`. Models in
  `core/models.py`; paths in `core/paths.py`; (de)serialize with
  `paths.write_model` / `read_model`.
- **Prompts are data, not logic**: all prompt text lives in `analyze/prompts.py`
  and `produce/formats/prompts.py`. Don't inline prompts in control flow.
- **LLM client** `core/llm.py`: OpenAI-compatible `POST {base_url}/chat/completions`
  (base_url is the FULL base **incl. /v1** — no auto-append). `temperature` is
  omitted unless `llm.temperature` is set (GPT-5 line rejects custom values).
- **Billing**: `core/billing.py` appends every LLM/TTS call to
  `data/<project>/cost.jsonl` via the `account(project_dir, stage)` contextvar;
  `ai-clip cost -p P` summarizes. Prices are USD, editable tables; MiMo TTS price is
  an unconfirmed estimate.
- **Config/env**: `core/config.py` loads `config/default.yaml` then `.env` (a tiny
  built-in loader). Provider key auto-detected from `llm.base_url` (deepseek→
  `DEEPSEEK_API_KEY`, openai→`OPENAI_API_KEY`). Overrides: `AICLIP_LLM_BASE_URL`,
  `AICLIP_LLM_MODEL`, `AICLIP_DATA_DIR`, `AICLIP_MPT_URL`, `AICLIP_COMFYUI_URL`.
- **CPU/GPU**: only transcription cares; `core/device.py` picks int8 (CPU) /
  float16 (GPU) for faster-whisper automatically.

## Setup / run / test

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"                      # core + tests
uv pip install -e ".[download,extract,llm]"     # heavy runtime extras (yt-dlp, faster-whisper)
ruff check src tests        # lint (must pass)
pytest -q                   # 102 tests, must stay green
```
Requires **ffmpeg + ffprobe** on PATH. `.env` already holds keys on the dev box:
`DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `MIMO_API_KEY`,
`PEXELS_API_KEY`, `TAVILY_API_KEY`. Default model `deepseek-v4-pro`
(base `https://api.deepseek.com/v1`); OpenAI = base `https://api.openai.com/v1` +
model `gpt-5.5`.

Scratch dir `./.testdata/` (gitignored) holds real-run projects: `dario` (24-min
YouTube → ~107s remix), `e2e` (80s clip), `mpttest`, `comfyui_test`.

## External services (state on the dev box)

- **DeepSeek** default LLM (verified). **gpt-5.5** verified via OpenAI.
- **MiMo TTS** (`https://api.xiaomimimo.com/v1`, `MIMO_API_KEY`) — voice clone works.
- **ComfyUI** for local CPU image gen: cloned at `E:/_Ai/comfyui`, run
  `python main.py --cpu`; model `sd_turbo.safetensors`; workflow
  `workflows/txt2img.json` (placeholder node text `AICLIP_PROMPT`). ~25s/image on CPU.
- **MoneyPrinterTurbo** at `E:/_Ai/mpt` (Docker image, run on host port 8081;
  `docker run -d --name mpt-api -p 127.0.0.1:8081:8080 -v //e/_Ai/mpt/config.toml:/MoneyPrinterTurbo/config.toml -v //e/_Ai/mpt/storage:/MoneyPrinterTurbo/storage ghcr.io/harry0703/moneyprinterturbo:latest python3 main.py`). `ai-clip mpt --theme T -p P` (set `AICLIP_MPT_URL=http://127.0.0.1:8081`). Verified 45s mp4.
- **NarratoAI** at `E:/_Ai/narrato` (own venv `.venv`). No HTTP API → `narrato.py`
  hard-wires `start_subclip_unified` via `_narrato_runner.py` in a subprocess.
  Backend auto-copies a CJK font into NarratoAI `resource/fonts/` (it ships none →
  Chinese subtitles were tofu boxes). Verified 15s combined.mp4.

## Coding standards

- Match existing style. ruff config in `pyproject.toml` (line length 100).
- Lazy-import heavy/optional deps (yt-dlp, faster-whisper) inside functions so the
  package imports without extras and tests stay light.
- Tests: mock external HTTP/yt-dlp; real ffmpeg integration tests skip if a tool/
  font is missing. Keep `pytest -q` green and `ruff check` clean before committing.
- Commit style: imperative subject + short body; end with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` (or your own trailer).
  Commit + push to `main` when a unit of work is done and verified.
- Bilingual README: `README.md` (中文 default) + `README-en.md`; update both.

## Pending work (priority order)

1. **`research` stage (MAIN next task)** — a human-in-the-loop deep-research step
   between `analyze` and `storyboard`. Decided provider: **Tavily**
   (`TAVILY_API_KEY` in `.env`). Build `research/` with: (a) LLM extracts key
   claims/queries from the transcript/analysis; (b) `search.py` provider calls
   Tavily; (c) LLM synthesizes findings into NEW viewpoints + citations →
   `data/<project>/research.json` + editable `research.md`. Add `ai-clip research
   -p P` (then human edits `research.md`); `storyboard` should read `research.json`
   and inject it via a new block in `produce/formats/prompts.py` (alongside
   `formula_block`/`intent_block`). Meter Tavily calls + LLM in billing. Mirror the
   `review.py` round-trip pattern for the human-editable file.
2. **A2 remix duration convergence** — remix targets `--duration` but real output
   overshoots (dario asked 60s, got ~107s). Tighten: instruct the LLM harder and/or
   trim/scale spans in `produce/formats/remix.py` so kept spans sum near target.
3. **review ergonomics** — `review.py` currently round-trips narration + remix
   timestamps only; also expose slideshow `caption` editing; optional duration-sum
   check on apply.
4. Optional/back-burner: storyboard review **UI** (thin Streamlit over `script.md`),
   an **agent layer** over `tools.py` (quality self-check/retry, one-shot "url→mp4").

## Gotchas

- **Windows**: ffmpeg drawtext can't handle drive-colon paths in filtergraphs — see
  `captions.py`/`assemble.py` (font copied into a tmp dir, ffmpeg run with `cwd`).
  In Git Bash, `/e/...` paths fail inside Python; use `E:/...`.
- **CJK fonts**: `core/fonts.py` auto-detects a system CJK font; caption/NarratoAI
  rendering depends on it (skips/falls back if none).
- **douyin/kuaishou**: yt-dlp cannot list `/user/` pages (verified) — discover
  degrades to a clear error; only direct video URLs work for download.
- **`.testdata/`, `data/`, `workflows/txt2img.json`, `.env`** are gitignored.
