# AGENTS.md — ai-clip 交接简介

短视频流水线编排器。串联开源工具,既能**二创**爆款(remix),也能从主题做**原创**。
Python 3.12,用 `uv` 管理。仓库:https://github.com/yiouyou/ai-clip(分支 `main`)。

本文件是给接手工作的自动化编码 agent 的交接说明。改代码前请通读。

## 它是什么(一段话)

ai-clip 是一个**轻量编排器**:负责数据契约、Typer CLI、设备适配、提示词、计费,
把重活外包给成熟项目(yt-dlp、faster-whisper、ffmpeg、ComfyUI)和 LLM/TTS 的 HTTP API。
默认流程是**提示词优先 + 人工介入**:各阶段产出人类可编辑的文件(`storyboard.md`、
`script.md`、prompt txt),人可以在不可逆的渲染之前介入。它**不是**单体应用——请保持这种风格。

## 流水线与文件地图

```
discover → download → extract → analyze → storyboard → [review] → (assets) → voiceover → assemble
```

| 阶段 | 模块 | 用的工具 | 说明 |
|------|------|---------|------|
| discover | `discover/`(youtube、bilibili、social=douyin/kuaishou、ranking) | yt-dlp | YT/B站支持关键词搜索;抖音/快手只能列用户主页,常不支持 → 降级为 NotImplementedError |
| download | `download/downloader.py` | yt-dlp | → `Clip` |
| extract | `extract/extractor.py`(+ `subtitles.py`、`export.py`) | ffmpeg + faster-whisper | `--subs` 用视频自带字幕代替 whisper |
| analyze | `analyze/`(analyzer + prompts) | LLM | 爆款拆解 → `ViralAnalysis`;`--intent info\|emotion\|sales` |
| storyboard | `produce/storyboard.py` + `produce/formats/*` | LLM | 按体裁 → `Storyboard`(若干 `Shot`) |
| review | `produce/review.py` | — | storyboard ↔ 可编辑 `script.md` 往返(文案人工编辑) |
| assets | `produce/assets/*`(factory、comfyui、prompt_only) | ComfyUI / 人工 | talking_head/slideshow/montage 的配图 |
| voiceover | `produce/voiceover.py` + `produce/tts/mimo.py` | MiMo TTS | 默认克隆源说话人音色 |
| assemble | `produce/assemble.py`(+ `captions.py`、`core/fonts.py`) | ffmpeg | 拼接、配音、可选烧字幕 |

编排:`pipeline.py`(各 `run_*` 函数,每个都开 `billing.account()` 块)。
组合工作流:`workflows.py`(transcribe/teardown/remix/original)。
为将来 agent 层准备的工具注册表:`tools.py`。
外部 produce 后端(可选,用于和自建对比):`produce/backends/`
(`moneyprinter.py` REST adapter、`narrato.py` 子进程硬接)。

### 视频体裁(`--format`)
`talking_head`(默认,口播 + 可选 b-roll 静图)、`slideshow`(图文卡 + 字幕 + 配音)、
`remix`(从**源视频**裁片段 + 新解说;不生成素材)、`montage`(全 AI 生成)。
四种体裁的提示词都在 `produce/formats/prompts.py`;format 模块只保留 LLM 输出→Shot 的映射。

## 硬约定(不要破坏)

- **文件名契约**:第 N 个镜头 ⇒ `assets/shot_NN.png` / `shot_NN.mp4`;remix 镜头带
  `source_start/source_end`,不需要素材。`assemble.check_assets` 以 `Shot.expected_files()` 为准。
- **每项目产物**在 `data/<project>/`:`clip.json`、`transcript.json`、`analysis.json`、
  `storyboard.json`(+ `.md`)、`script.md`、`prompts/`、`assets/`、`voice/`、
  `candidates.json`、`cost.jsonl`、`output.mp4`。模型在 `core/models.py`;路径在
  `core/paths.py`;用 `paths.write_model` / `read_model` 读写。
- **提示词是数据,不是逻辑**:所有提示词文本在 `analyze/prompts.py` 和
  `produce/formats/prompts.py`。不要把提示词内联进控制流。
- **LLM 客户端** `core/llm.py`:OpenAI 兼容 `POST {base_url}/chat/completions`
  (base_url 是**含 /v1 的完整基址**——不自动拼接)。除非设了 `llm.temperature`,
  否则不发 `temperature`(GPT-5 系列拒绝自定义值)。
- **计费**:`core/billing.py` 通过 `account(project_dir, stage)` contextvar 把每次
  LLM/TTS 调用追加到 `data/<project>/cost.jsonl`;`ai-clip cost -p P` 汇总。价格是 USD
  可编辑表;MiMo TTS 价格是未确认的估算。
- **配置/env**:`core/config.py` 先读 `config/default.yaml` 再读 `.env`(内置极简 loader)。
  provider key 按 `llm.base_url` 自动识别(deepseek→`DEEPSEEK_API_KEY`,
  openai→`OPENAI_API_KEY`)。覆盖项:`AICLIP_LLM_BASE_URL`、`AICLIP_LLM_MODEL`、
  `AICLIP_DATA_DIR`、`AICLIP_MPT_URL`、`AICLIP_COMFYUI_URL`。
- **CPU/GPU**:只有转写在意;`core/device.py` 自动为 faster-whisper 选 int8(CPU)/
  float16(GPU)。

## 安装 / 运行 / 测试

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"                      # 核心 + 测试
uv pip install -e ".[download,extract,llm]"     # 运行时重依赖(yt-dlp、faster-whisper)
ruff check src tests        # lint(必须通过)
pytest -q                   # 179 个测试,须保持全绿
```
需要 PATH 上有 **ffmpeg + ffprobe**。开发机的 `.env` 已配:`DEEPSEEK_API_KEY`、
`OPENAI_API_KEY`、`GEMINI_API_KEY`、`MIMO_API_KEY`、`PEXELS_API_KEY`、`TAVILY_API_KEY`。
默认模型 `deepseek-v4-pro`(base `https://api.deepseek.com/v1`);OpenAI = base
`https://api.openai.com/v1` + 模型 `gpt-5.5`。

临时目录 `./.testdata/`(已 gitignore)放真跑项目:`dario`(24 分钟 YouTube → ~107s
remix)、`e2e`(80s 片)、`mpttest`、`comfyui_test`。

## 外部服务(开发机现状)

- **DeepSeek** 默认 LLM(已验证)。**gpt-5.5** 经 OpenAI 已验证。
- **MiMo TTS**(`https://api.xiaomimimo.com/v1`,`MIMO_API_KEY`)——声音克隆可用。
- **ComfyUI** 本地 CPU 出图:克隆在 `E:/_Ai/comfyui`,运行 `python main.py --cpu`;
  模型 `sd_turbo.safetensors`;工作流 `workflows/txt2img.json`(占位节点文本
  `AICLIP_PROMPT`)。CPU 单张约 25s。
- **MoneyPrinterTurbo** 在 `E:/_Ai/mpt`(Docker 镜像,映射到宿主 8081:
  `docker run -d --name mpt-api -p 127.0.0.1:8081:8080 -v //e/_Ai/mpt/config.toml:/MoneyPrinterTurbo/config.toml -v //e/_Ai/mpt/storage:/MoneyPrinterTurbo/storage ghcr.io/harry0703/moneyprinterturbo:latest python3 main.py`)。`ai-clip mpt --theme T -p P`(设 `AICLIP_MPT_URL=http://127.0.0.1:8081`)。已验证出 45s mp4。
- **NarratoAI** 在 `E:/_Ai/narrato`(自带 venv `.venv`)。无 HTTP API → `narrato.py`
  通过 `_narrato_runner.py` 子进程硬接其 `start_subclip_unified`。后端会自动把中文字体
  复制进 NarratoAI 的 `resource/fonts/`(它不带字体 → 中文字幕原本是豆腐块)。已验证出
  15s combined.mp4。

## 编码规范

- 与现有风格一致。ruff 配置在 `pyproject.toml`(行宽 100)。
- 重/可选依赖(yt-dlp、faster-whisper)在函数内**懒导入**,使包在不装 extras 时也能 import、
  测试更轻。
- 测试:mock 外部 HTTP/yt-dlp;真实 ffmpeg 集成测试在缺工具/字体时跳过。提交前保持
  `pytest -q` 全绿、`ruff check` 干净。
- 提交风格:祈使句标题 + 简短正文;结尾加
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`(或你自己的 trailer)。
  一个工作单元完成并验证后 commit + push 到 `main`。
- 双语 README:`README.md`(中文默认)+ `README-en.md`;两版都要更新。

## 待办(按优先级)

1. **`research` 阶段(主要的下一步)** —— 在 `analyze` 和 `storyboard` 之间加一个
   人工介入的深度研究步骤。已定 provider:**Tavily**(`.env` 里有 `TAVILY_API_KEY`)。
   建 `research/`:(a) LLM 从文稿/分析里抽关键论点/查询;(b) `search.py` provider 调
   Tavily;(c) LLM 把结果综合成**新观点 + 引用** → `data/<project>/research.json` +
   可编辑的 `research.md`。加 `ai-clip research -p P`(之后人工编辑 `research.md`);
   `storyboard` 应读取 `research.json` 并通过 `produce/formats/prompts.py` 里新增的一个块
   注入(和 `formula_block`/`intent_block` 并列)。Tavily 调用 + LLM 计入 billing。
   人工可编辑文件照 `review.py` 的往返模式做。
2. **A2 remix 时长收敛** —— remix 以 `--duration` 为目标,但实际超时(dario 要 60s 出
   ~107s)。收紧:给 LLM 更强约束,且/或在 `produce/formats/remix.py` 里裁/缩片段使总时长
   贴近目标。
3. **review 体验** —— `review.py` 目前只往返解说 + remix 时间戳;再支持 slideshow 的
   `caption` 编辑;apply 时可加总时长校验。
4. 可选/次要:storyboard 审片 **UI**(基于 `script.md` 的薄 Streamlit);基于 `tools.py` 的
   **agent 层**(质量自检/重试,一句话 url→mp4)。

## 坑

- **Windows**:ffmpeg drawtext 在 filtergraph 里处理不了盘符冒号路径——见
  `captions.py`/`assemble.py`(字体复制进临时目录、ffmpeg 用 `cwd` 运行)。Git Bash 里
  `/e/...` 路径在 Python 内会失败,用 `E:/...`。
- **Windows pid 检测**:不要用 `os.kill(pid, 0)` 判断进程是否存在。Unix 上这是探测,
  但 Windows 上可能终止目标进程。`RadarRunLock` 这类锁应使用 WinAPI
  `OpenProcess`/`CloseHandle` 或等价安全方法;此前 targeted pytest 自杀就是这个坑。
- **Codex 插件稳定性**:若 Codex 崩溃并提示
  `interface.defaultPrompt[0]: prompt must be at most 128 characters`,先检查
  `C:/Users/Administrator/.codex/.tmp/.../.codex-plugin/plugin.json`。本机曾因
  `ngs-analysis` 插件 defaultPrompt 过长触发 warning;临时修短后可恢复,但插件刷新可能覆盖。
  `config.toml` 可能无 WSL 字段;若仍提示兼容性问题,去 Codex 设置里关闭“在 WSL 中运行”。
- **中文字体**:`core/fonts.py` 自动探测系统中文字体;字幕/NarratoAI 渲染依赖它
  (没有就跳过/回退)。
- **抖音/快手**:yt-dlp 无法列 `/user/` 主页(已验证)——discover 降级为清晰报错;
  只有直链视频能下载。
- **`.testdata/`、`data/`、`workflows/txt2img.json`、`.env`** 都已 gitignore。
