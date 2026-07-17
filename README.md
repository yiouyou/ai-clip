# ai-clip

[English](README-en.md) | **中文**

ai-clip 是一个把开源工具串成短视频流水线的**轻量编排器**，既能二创爆款，也能从主题做原创。
它负责数据契约、CLI、设备适配、提示词、计费和流程状态；下载、转写、生成、合成等重活交给
yt-dlp、faster-whisper、ComfyUI、ffmpeg、LLM/TTS API 等成熟工具。

```text
discover -> download -> extract -> analyze -> research -> storyboard -> review -> assets -> voiceover -> assemble
```

详细文档:

- [工作流说明](docs/workflows.md)
- [Research 阶段](docs/research.md)
- [Daily Radar 选题流程](docs/radar.md)
- [Pair Review](docs/pair-review.md)

## 设计原则

- **提示词优先 + 人工介入**: `storyboard.md`、`script.md`、`research.md` 都是可编辑文件。
- **文件名契约稳定**: 第 N 个镜头素材写入 `assets/shot_NN.png` / `shot_NN.mp4`。
- **媒体增量缓存**: 系统图片和语音带 manifest；prompt、模型、声音或参考音频未变时不会重复生成。
- **项目产物独立**: 每个项目都在 `data/<project>/` 下，可单独重跑阶段。
- **轻量 metadata**: 关键产物写 `<artifact>.meta.json`，用于判断 `fresh/stale/missing`。
- **可观测流程**: 组合 workflow 写 `data/<project>/runs/<workflow>.json`。
- **能力可组合**: 视频内容获取、research 和 review 使用共享引擎，workflow 只保留领域适配。
- **执行契约统一**: workflow 阶段使用显式 invocation/result envelope，业务函数保持强类型参数。

## 环境要求

- Python 3.12+ 和 [uv](https://github.com/astral-sh/uv)
- PATH 上有 **ffmpeg + ffprobe**
- 可选: OpenAI-compatible LLM key、Tavily key、MiMo TTS key、本地 ComfyUI

## 安装

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv pip install -e ".[download,extract]"
cp .env.example .env
```

## 快速上手

二创:

```bash
ai-clip remix "<clip-url>" -p demo --theme "用 60 秒讲清复利"
ai-clip status -p demo
ai-clip assemble -p demo
```

原创:

```bash
ai-clip original -p promo --theme "城市夜骑 vlog 开场" --shots 5
ai-clip original -p promo --theme "AI 公司的生态位竞争" --research
```

`original --research` 使用不依赖源视频的主题研究；`source-draft` 和 `remix` 使用 transcript/
analysis 驱动的源视频研究。原创流程缺少必要图片时会停在 `waiting`，不会提前调用 TTS。

单视频原创口播:

```bash
ai-clip source-draft "<clip-url>" -p demo --research --theme "用复杂系统解释这个事件"
ai-clip source-draft "<clip-url>" -p demo --whisper-model small   # 长视频 CPU 转写更快
```

项目级研究:

```bash
ai-clip research -p demo --theme "AI 公司的生态位竞争" --max-searches 2
ai-clip storyboard -p demo --theme "AI 公司的生态位竞争"
```

每日选题:

```bash
ai-clip daily-radar --top 3
ai-clip radar-feedback accept --date 2026-07-09 --reason "选题角度合适"
```

## 常用命令

| 命令 | 用途 |
|------|------|
| `ai-clip transcribe <url> -p P` | 下载、转写并导出 `.srt` / `.txt` |
| `ai-clip teardown <url> -p P` | 下载、转写并拆解爆款公式 |
| `ai-clip source-draft <url> -p P` | 单视频生成原创口播稿;默认只复用与本次 URL、参数、模型及上游输入匹配的产物,`--no-resume` 可强制重跑 |
| `ai-clip research -p P --theme T` | 生成 `research.json` 和可编辑 `research.md` |
| `ai-clip storyboard -p P --theme T` | 生成分镜、素材 prompt 和 `storyboard.md` |
| `ai-clip review -p P` / `--apply` | `storyboard.json` 和 `script.md` 往返 |
| `ai-clip pair-review -p P --artifact script --rewrite` | 多模型互审、单次改写并验证 |
| `ai-clip status -p P` | 查看 artifact freshness 和素材缺失 |
| `ai-clip status -p P --json` | 输出机器可读项目状态 |
| `ai-clip assets -p P` | 生成缺失或已过期的图片素材，保留无 manifest 的人工图片 |
| `ai-clip voiceover -p P` | 按镜头增量生成配音，保留无 manifest 的人工 WAV |
| `ai-clip assemble -p P` | 合成最终 mp4 |
| `ai-clip doctor` | 本地环境诊断 |
| `ai-clip radar-feedback accept\|reject --date D` | 记录明确选题反馈，用于后续排序校准 |
| `ai-clip run-status --workflow W -p P --json` | 查看一次运行的阶段、产物、历史目录和 API 用量 |

## 视频体裁

`storyboard`、`remix`、`original` 支持 `--format`:

| 体裁 | 产出 | 素材需求 |
|------|------|----------|
| `talking_head` | 逐句口播 + 可选 b-roll | 可选静图 |
| `slideshow` | 图文卡 + 屏幕字幕 + 配音 | 每张卡一张图 |
| `remix` | 源视频裁片段 + 新解说 | 无，使用源视频 |
| `montage` | 全 AI 生成多镜头短剧 | 每镜头图/视频 |

## 配置

配置加载顺序:

1. `config/default.yaml`
2. `.env`
3. 命令行参数

YAML 配置使用严格校验：未知字段、负数 Top N、零 worker 等无效值会直接报错，不会静默
回退到默认配置。

常用环境变量:

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

`NEWAPP_URL` 应填写 OpenAI-compatible 完整基址,包含 `/v1`。

## 本地检查

不触发付费 API 的 Windows smoke:

```powershell
.\scripts\smoke.ps1
```

提交前建议:

```bash
uv run ruff check src tests
uv run --extra download pytest -q
```

## 可选后端

- **ComfyUI**: 本地自动出图，配置 `assets.image_provider: comfyui` 或 `auto`。
- **Smart Illustrator**: 信息图/封面风格图片 provider，见配置里的 `assets.smart_illustrator_*`。
- **MoneyPrinterTurbo / NarratoAI**: 作为外部 produce 后端用于对比效果。

## 成本

LLM/TTS 调用会追加到 `data/<project>/cost.jsonl`:

```bash
ai-clip cost -p demo
```

## 许可与依赖

ai-clip 编排第三方项目，包括 yt-dlp、faster-whisper、ComfyUI、ffmpeg 等。再分发或商用前，
请分别查看这些项目的许可证。
