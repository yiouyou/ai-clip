# ai-clip

An orchestrator that chains open-source tools into a short-video pipeline for both
**remixing** a viral clip (二创) and producing **original** videos from a theme.

```
download → extract → analyze → storyboard → (human / ComfyUI makes assets) → voiceover → assemble
 yt-dlp     ffmpeg +   LLM       LLM prompts                                    MiMo TTS    ffmpeg
            whisper   teardown                                                  (clone)
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

## Video formats (`--format`)

`storyboard` / `remix` / `original` accept `--format` to match the viral archetype:

| Format | What it produces | Assets needed |
|--------|------------------|---------------|
| `talking_head` (default) | Narration lines + optional b-roll stills | Optional b-roll images |
| `slideshow` | Image cards + on-screen captions + narration | One image per card |
| `remix` | Spans cut from the **source clip** + new narration | None (uses source) |
| `montage` | Fully AI-generated multi-shot drama | Image (+video) per shot |

`remix` needs a source clip (use `ai-clip remix <url>`); it produces a playable
video with just `voiceover` + `assemble` — no manual asset creation.

Run any stage on its own: `discover`, `download`, `extract`, `export`,
`analyze`, `storyboard`, `status`, `voiceover`, `assemble`.

## Composed workflows

One command chains the stages end to end:

| Command | Flow | Output |
|---------|------|--------|
| `ai-clip transcribe <url> -p P` | download → extract → export | `.srt` + `.txt` |
| `ai-clip teardown <url> -p P` | download → extract → analyze | viral formula |
| `ai-clip remix <url> --theme T -p P` | download → extract → analyze → remix storyboard → cloned voiceover → assemble | **`output.mp4` (fully auto)** |
| `ai-clip original --theme T -p P [-f talking_head\|slideshow]` | storyboard → assets (ComfyUI if available) → voiceover → assemble | `output.mp4`, or a prompt to fill `assets/` if no local generator |

`discover` finds source URLs to feed these: `ai-clip discover "AI" -p P --top 5`.
YouTube/Bilibili support keyword search; douyin/kuaishou have no search API, so pass
`--platform douyin --channel <user page URL>` to rank a creator's recent clips (or
just hand a direct video URL to `ai-clip download`).

### Intent (info / emotion / sales)

`--intent` steers analyze + storyboard:

- `info` (default) — knowledge-first, neutral explainer.
- `emotion` — an opinionated take: extract a **stance + emotional charge** from the
  news/event and express it (not neutral reporting). Optional `--stance "..."` to
  fix the angle; otherwise the LLM picks one.
- `sales` — product promo (pain → agitate → product → proof → CTA). Pass a reusable
  product profile with `--product products/mine.yaml` (see `products/*.example.yaml`).

```bash
ai-clip remix <url> --theme "锐评本周AI" --intent emotion -p p1
ai-clip original --theme "麻友手气" --intent sales --product products/mahjong.yaml -p p2
```

### Burned-in captions

Add `--captions` (or `burn_captions: true` in config) to burn each shot's caption
(slideshow) or narration (talking-head/remix) into the video via ffmpeg `drawtext`.
A CJK-capable system font is auto-detected; if none is found, burning is skipped.

```bash
ai-clip remix <url> --theme T --captions -p p1
ai-clip assemble -p p1 --captions
```

## Voiceover (MiMo TTS + voice cloning)

`ai-clip voiceover` synthesizes each shot's narration with Xiaomi
[MiMo-V2.5-TTS](https://mimo.mi.com). With `tts.clone_from_source: true` (default)
and the `mimo-v2.5-tts-voiceclone` model, it cuts a short reference snippet from
the **voice extracted in the `extract` stage** and clones the original speaker —
so a remix can keep the source creator's timbre. Without a source clip it falls
back to a preset voice (`tts.voice`). Set `MIMO_API_KEY` in `.env`.

```bash
ai-clip voiceover --project demo     # -> data/demo/voice/shot_NN.wav
ai-clip assemble  --project demo     # picks up voice/ automatically
```

## ComfyUI (optional, local auto image generation)

1. Run ComfyUI (default `http://127.0.0.1:8188`).
2. Export a workflow in **API format**, set the positive-prompt node's `text` to the
   literal `AICLIP_PROMPT`, and save it as `workflows/txt2img.json`
   (see `workflows/txt2img.example.json`).
3. Set `assets.image_provider: auto` (default). When ComfyUI answers and the
   workflow exists, images are generated automatically; otherwise ai-clip falls back
   to prompt-only (you create assets in the browser).

**No GPU? Still works.** The fastest CPU model is **SD-Turbo** (SD1.5, 1 step):
launch ComfyUI with `--cpu`, drop `sd_turbo.safetensors` in `models/checkpoints/`,
and use the bundled `workflows/txt2img.json` (512×768, 1 step). Verified: ~25s per
image on CPU. Text-to-**video** is not practical on CPU — keep `video_provider:
prompt_only` (即梦/可灵 in browser) or use a GPU. talking_head/slideshow only need
stills, so CPU ComfyUI fully covers them; remix needs no generation at all.

## Docker

```bash
cp .env.example .env
docker compose --profile cpu up        # CPU box
docker compose --profile gpu up        # GPU box (whisper on CUDA + ComfyUI service)
```

`profiles` control what gets installed/started; the orchestrator and each heavy tool
get their own container and communicate over a shared `./data` volume.

## External produce backends (optional)

Alternatives to the self-built storyboard→voiceover→assemble path, for comparison:

- **MoneyPrinterTurbo** (theme → stock-footage + TTS + subtitle video). It has a
  REST API, so `ai-clip mpt --theme T -p P` drives it cleanly. Run MoneyPrinterTurbo
  (Docker) with an LLM + Pexels/Pixabay key, set `AICLIP_MPT_URL` (default
  `http://127.0.0.1:8080`). Verified end-to-end.
- **NarratoAI** (解说二创). It is WebUI-only (no HTTP API), so `NarratoBackend`
  hard-wires its internal `start_subclip_unified` via a subprocess runner inside
  NarratoAI's own repo/venv; an ai-clip remix storyboard maps to its clip script.

These are opt-in; the default pipeline needs none of them.

## License & dependencies

ai-clip orchestrates third-party projects (yt-dlp, faster-whisper, ComfyUI, etc.);
review each project's license before redistribution or commercial use.
