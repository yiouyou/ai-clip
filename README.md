# ai-clip

An orchestrator that chains open-source tools into a short-video pipeline for both
**remixing** a viral clip (二创) and producing **original** videos from a theme.

```
download → extract → analyze → storyboard → (human / ComfyUI makes assets) → assemble
 yt-dlp     ffmpeg +   LLM       LLM prompts                                    ffmpeg
            whisper   teardown
```

ai-clip is a lightweight **orchestrator**: it owns the data contract, the CLI, and
device adaptation, and delegates heavy work to mature projects (yt-dlp,
faster-whisper) and to image/video generation you run yourself (ComfyUI locally, or
即梦 / Gemini in the browser with a plan you already pay for).

## Design

- **Prompt-first, human-in-the-loop produce.** Generation APIs are off by default.
  The storyboard step writes per-shot prompts; you create assets on a website (or
  via local ComfyUI) and drop them into `assets/`. The **filename contract**
  (`shot_NN.png` / `shot_NN.mp4`) means assemble doesn't care how an asset was made
  — ComfyUI API and human downloads can be freely mixed.
- **CPU/GPU adaptive.** The only GPU-sensitive stage is transcription;
  faster-whisper uses `int8` on CPU and `float16` on GPU automatically.
- **Per-project artifacts.** Everything for a project lives under
  `data/<project>/` as JSON + files, so any stage can be re-run in isolation.

## Requirements

- Python 3.12+ and [uv](https://github.com/astral-sh/uv)
- **ffmpeg + ffprobe** on PATH (`winget install ffmpeg` / `apt install ffmpeg`)
- Optional: an OpenAI-compatible LLM key (DeepSeek/Qwen/…); a local ComfyUI

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"            # core + tests
uv pip install -e ".[download,extract,llm]"   # add the heavy runtime extras
cp .env.example .env                  # fill in your LLM key
```

## Usage

Remix a viral clip into a new video on your own theme:

```bash
ai-clip remix "<clip-url>" --project demo --theme "用 60 秒讲清复利"
# -> writes data/demo/storyboard.md + data/demo/prompts/*.txt
# create each asset, save into data/demo/assets/ using the contract filenames
ai-clip status   --project demo       # see which shots still lack assets
ai-clip assemble --project demo       # -> data/demo/output.mp4
```

Original (theme only, no source clip):

```bash
ai-clip original --project promo --theme "城市夜骑 vlog 开场" --shots 5
```

Run any stage on its own: `download`, `extract`, `analyze`, `storyboard`,
`status`, `assemble`.

## ComfyUI (optional, local auto image generation)

1. Run ComfyUI (default `http://127.0.0.1:8188`).
2. Export a workflow in **API format**, set the positive-prompt node's `text` to the
   literal `AICLIP_PROMPT`, and save it as `workflows/txt2img.json`
   (see `workflows/txt2img.example.json`).
3. Set `assets.image_provider: auto` (default). When ComfyUI answers and the
   workflow exists, images are generated automatically; otherwise ai-clip falls back
   to prompt-only (you create assets in the browser).

## Docker

```bash
cp .env.example .env
docker compose --profile cpu up        # CPU box
docker compose --profile gpu up        # GPU box (whisper on CUDA + ComfyUI service)
```

`profiles` control what gets installed/started; the orchestrator and each heavy tool
get their own container and communicate over a shared `./data` volume.

## License & dependencies

ai-clip orchestrates third-party projects (yt-dlp, faster-whisper, ComfyUI, etc.);
review each project's license before redistribution or commercial use.
