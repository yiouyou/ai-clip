# ai-clip

[English](README-en.md) | **中文**

一个把开源工具串成短视频流水线的**编排器**,既能**二创**(复刻爆款),也能从主题做**原创**。

```
discover → download → extract → analyze → storyboard →(人工/ComfyUI 出素材)→ voiceover → assemble
 选题       yt-dlp     ffmpeg+    LLM       LLM 提示词                          MiMo TTS    ffmpeg
            爆款       whisper    拆解                                          (克隆)
```

ai-clip 本身是个轻量编排器:负责数据契约、CLI、设备适配,把重活外包给成熟项目
(yt-dlp、faster-whisper)以及你自己掌控的图像/视频生成(本地 ComfyUI,或在浏览器里用
你已购的即梦 / Gemini)。

## 设计原则

- **提示词优先 + 人工介入(human-in-the-loop)**:生成类 API 默认关闭。storyboard 阶段
  产出每镜头提示词;你在网站(或本地 ComfyUI)生成素材,丢进 `assets/`。**文件名契约**
  (`shot_NN.png` / `shot_NN.mp4`)让 assemble 不关心素材怎么来的——ComfyUI 自动出图和
  人工下载可以随意混用。
- **CPU/GPU 自适应**:唯一吃 GPU 的是转写;faster-whisper 在 CPU 上用 `int8`、GPU 上自动用
  `float16`。
- **每项目独立产物**:一个项目的所有产物都在 `data/<project>/` 下(JSON + 文件),任意阶段
  可单独重跑。

## 环境要求

- Python 3.12+ 和 [uv](https://github.com/astral-sh/uv)
- PATH 上有 **ffmpeg + ffprobe**(`winget install ffmpeg` / `apt install ffmpeg`)
- 可选:一个 OpenAI 兼容的 LLM key(DeepSeek/Qwen/…);本地 ComfyUI

## 安装

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"            # 核心 + 测试
uv pip install -e ".[download,extract,llm]"   # 加上运行时重依赖
cp .env.example .env                  # 填入你的 LLM key
```

## 快速上手

把一条爆款二创成你自己主题的新视频:

```bash
ai-clip remix "<clip-url>" --project demo --theme "用 60 秒讲清复利"
# -> 生成 data/demo/storyboard.md + data/demo/prompts/*.txt
# 按文件名契约把生成的素材存进 data/demo/assets/
ai-clip status   --project demo       # 看还缺哪些镜头的素材
ai-clip assemble --project demo       # -> data/demo/output.mp4
```

原创(只给主题,无源视频):

```bash
ai-clip original --project promo --theme "城市夜骑 vlog 开场" --shots 5
```

## 视频体裁(`--format`)

`storyboard` / `remix` / `original` 都支持 `--format` 匹配不同爆款形态:

| 体裁 | 产出什么 | 需要的素材 |
|------|---------|-----------|
| `talking_head`(默认) | 逐句口播 + 可选 b-roll 配图 | 可选 b-roll 图 |
| `slideshow` | 图文卡 + 屏幕字幕 + 配音 | 每张卡一张图 |
| `remix` | 从**源视频**裁片段 + 新解说 | 无(用源视频) |
| `montage` | 全 AI 生成的多镜头短剧 | 每镜头图(+视频) |

`remix` 需要源视频(用 `ai-clip remix <url>`),只需 `voiceover` + `assemble` 即可出片,
无需人工做素材。

任意单步也可单独运行:`discover`、`download`、`extract`、`export`、`analyze`、
`storyboard`、`status`、`voiceover`、`assemble`。

## 组合工作流

一条命令端到端串起各阶段:

| 命令 | 流程 | 产出 |
|------|------|------|
| `ai-clip transcribe <url> -p P` | download → extract → export | `.srt` + `.txt` |
| `ai-clip teardown <url> -p P` | download → extract → analyze | 爆款公式 |
| `ai-clip remix <url> --theme T -p P` | download → extract → analyze → remix 分镜 → 克隆配音 → assemble | **`output.mp4`(全自动)** |
| `ai-clip original --theme T -p P [-f talking_head\|slideshow]` | 分镜 → 素材(有 ComfyUI 则自动)→ 配音 → assemble | `output.mp4`,无本地生成器时提示去填 `assets/` |

`discover` 帮这些找源 URL:`ai-clip discover "AI" -p P --top 5`。YouTube/B站 支持关键词
搜索;抖音/快手无搜索 API,请用 `--platform douyin --channel <用户主页URL>` 对某作者的
近期作品排序(或直接把视频链接交给 `ai-clip download`)。

### 意图(info / emotion / sales)

`--intent` 同时影响 analyze 和 storyboard:

- `info`(默认)—— 知识优先,中立讲解。
- `emotion` —— **带立场的观点表达**:从新闻/事件里提炼**立场 + 情绪**并表达出来(而非中立
  报道)。可用 `--stance "..."` 指定立场,不指定则由 LLM 自己挑。
- `sales` —— 带货(痛点 → 煽动 → 产品 → 证言 → CTA)。用 `--product products/mine.yaml`
  传入可复用的产品档案(见 `products/*.example.yaml`)。

```bash
ai-clip remix <url> --theme "锐评本周AI" --intent emotion -p p1
ai-clip original --theme "麻友手气" --intent sales --product products/mahjong.yaml -p p2
```

### 字幕烧录

加 `--captions`(或在 config 里 `burn_captions: true`),用 ffmpeg `drawtext` 把每镜头的
字幕(slideshow)或旁白(口播/remix)烧进画面。会自动探测可用的中文字体,找不到则跳过烧录。

```bash
ai-clip remix <url> --theme T --captions -p p1
ai-clip assemble -p p1 --captions
```

## 配音(MiMo TTS + 声音克隆)

`ai-clip voiceover` 用小米 [MiMo-V2.5-TTS](https://mimo.mi.com) 合成每镜头旁白。当
`tts.clone_from_source: true`(默认)且用 `mimo-v2.5-tts-voiceclone` 模型时,它会从
**extract 阶段提取的人声**里截一小段参考音频克隆原说话人音色——这样二创能保留原作者的
声线。没有源视频时回退到预设音色(`tts.voice`)。在 `.env` 里设 `MIMO_API_KEY`。

```bash
ai-clip voiceover --project demo     # -> data/demo/voice/shot_NN.wav
ai-clip assemble  --project demo     # 自动带上 voice/
```

## ComfyUI(可选,本地自动出图)

1. 启动 ComfyUI(默认 `http://127.0.0.1:8188`)。
2. 导出 **API 格式**的工作流,把正向提示词节点的 `text` 设为字面量 `AICLIP_PROMPT`,
   存为 `workflows/txt2img.json`(参考 `workflows/txt2img.example.json`)。
3. 设 `assets.image_provider: auto`(默认)。ComfyUI 可达且工作流存在时自动出图;否则回退到
   prompt-only(你去浏览器生成)。

**没有 GPU 也能跑。** CPU 上最快的模型是 **SD-Turbo**(SD1.5,1 步):用 `--cpu` 启动
ComfyUI,把 `sd_turbo.safetensors` 放进 `models/checkpoints/`,用自带的
`workflows/txt2img.json`(512×768,1 步)。实测 CPU 单张约 25s。文生**视频**在 CPU 上不
实用——视频继续走 `video_provider: prompt_only`(浏览器即梦/可灵)或用 GPU。
talking_head/slideshow 只需静图,CPU ComfyUI 完全够;remix 根本不需要生成。

## Docker

```bash
cp .env.example .env
docker compose --profile cpu up        # CPU 机
docker compose --profile gpu up        # GPU 机(whisper 走 CUDA + ComfyUI 服务)
```

`profiles` 控制装/起哪些;编排器和每个重型工具各自独立容器,通过共享的 `./data` 卷通信。

## 外部 produce 后端(可选)

自建 storyboard→voiceover→assemble 路线之外的替代方案,用于对比效果。
(`mpt` 只是 MoneyPrinterTurbo 的简称。)

- **MoneyPrinterTurbo**(主题 → 库存素材 + TTS + 字幕成片)。它有 REST API,所以
  `ai-clip mpt --theme T -p P` 能干净驱动。用 Docker 起 MoneyPrinterTurbo(配 LLM +
  Pexels/Pixabay key),设 `AICLIP_MPT_URL`(默认 `http://127.0.0.1:8080`)。已端到端验证。
- **NarratoAI**(解说二创)。它只有 WebUI(无 HTTP API),所以 `NarratoBackend` 通过子进程
  在 NarratoAI 自己的 repo/venv 里硬接其内部函数 `start_subclip_unified`;ai-clip 的 remix
  分镜会映射成它的剪辑脚本。

这些都是可选的;默认流水线一个都不需要。

## 许可与依赖

ai-clip 编排了第三方项目(yt-dlp、faster-whisper、ComfyUI 等);再分发或商用前请查看各项目
各自的许可证。
