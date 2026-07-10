# 工作流说明

ai-clip 的普通项目流程围绕 `data/<project>/` 组织。每个阶段都可以单独跑，也可以通过组合
workflow 串起来。

阶段和 workflow 的唯一目录在 `src/ai_clip/registry.py`。外部名称统一使用 kebab-case；
Python 函数和 agent tool 名使用 snake_case。注册表同时提供阶段说明、输入输出、runner、
tool schema、CLI 名称，以及 workflow 的顺序和可选条件。Typer 参数函数保持显式定义，
因此 `--help` 和类型签名仍可直接检查。

共享能力与领域适配器分开：`extract/remote.py` 提供 URL 字幕、音频下载、Whisper 兜底和缓存；
`research_engine.py` 统一查询解析、角度配额、最多 3 次搜索及结果整理。项目 research 与
Daily Radar research 只定义各自的素材和编辑角度。Pair Review 接收中性的 `ArtifactRef`，
由 `artifact_catalog.py` 解析项目或 Radar 路径，因此 review 引擎不依赖具体 workflow 布局。

## 单步阶段

常用阶段:

```bash
ai-clip discover "AI" -p demo --top 5
ai-clip download "<url>" -p demo
ai-clip extract -p demo --subs
ai-clip analyze -p demo --intent info
ai-clip research -p demo --theme "AI 公司的生态位竞争" --max-searches 2
ai-clip storyboard -p demo --theme "AI 公司的生态位竞争"
ai-clip review -p demo
ai-clip review -p demo --apply
ai-clip assets -p demo
ai-clip voiceover -p demo
ai-clip assemble -p demo
```

`status` 同时检查关键文本产物 freshness 和分镜素材缺失:

```bash
ai-clip status -p demo
ai-clip status -p demo --json
```

JSON 输出统一使用 `schema_version / command / status / result` envelope。`result` 包含
`research/storyboard/source_draft` 的 freshness、review/rewrite/verify 链、分镜镜头数、
缺失素材、运行状态和本次 API usage。

## 组合 workflow

| 命令 | 流程 | 产出 |
|------|------|------|
| `ai-clip transcribe <url> -p P` | download -> extract -> export | `.srt` + `.txt` |
| `ai-clip teardown <url> -p P` | download -> extract -> analyze | 爆款公式 |
| `ai-clip source-draft <url> -p P` | download -> extract -> analyze -> [research] -> source_draft | `source_draft.md` |
| `ai-clip remix <url> --theme T -p P` | download -> extract -> analyze -> [research] -> remix storyboard -> voiceover -> assemble | `output.mp4` |
| `ai-clip original --theme T -p P` | [research] -> storyboard -> assets -> voiceover -> assemble | `output.mp4` 或提示补素材 |

这些顺序由 `WorkflowSpec` 执行，不再分别硬编码在 CLI、tools 和 Radar 中。当前声明如下：

```text
transcribe   = download -> extract -> export
teardown     = download -> extract -> analyze
source-draft = download -> extract -> analyze -> [research] -> source-draft
remix        = download -> extract -> analyze -> [research] -> storyboard -> voiceover -> assemble
original     = [research] -> storyboard -> assets -> voiceover -> assemble
daily-radar  = collect -> zack-ranking -> source-content -> content-rerank -> zack-selection
               -> [source-research] -> zack-draft -> [pair-review]
               -> [pair-rewrite -> pair-verify]
```

`source-draft`、`remix`、`original` 都可显式启用项目级 research:

```bash
ai-clip source-draft "<url>" -p demo --research --theme "..." --research-searches 1
ai-clip remix "<url>" -p demo --theme "..." --research --research-searches 1
ai-clip original -p demo --theme "..." --research --research-searches 1
```

`source-draft` 默认复用有效的 `clip.json`、`transcript.json`、`analysis.json`、
`research.md` 和 `source_draft.md`。有效性会核对 URL、阶段参数、模型和上游文件；任一上游
重跑后，下游不会继续复用。旧项目中可确认同一来源的 clip/transcript/analysis 会补写
metadata，无法确认生成参数的旧 research/draft 会重跑一次。需要强制全量重跑时使用
`--no-resume`。长视频在 CPU 上转写较慢时，可用 `--whisper-model small` 临时降低本次转写模型。

`remix`、`original` 未启用 `--research` 时不会隐式读取目录中的旧 `research.md`；启用后也只
注入与当前主题、模型和上游输入匹配的 research。Daily Radar 的 draft 同样通过 research
manifest 判断是否可用，而不是只依赖运行状态标签。

组合 workflow 会写运行状态:

```text
data/<project>/runs/<workflow>.json
```

每次重新启动组合 workflow 会生成新的 `run_id` 和递增的 attempt；上一轮状态归档在
`data/<project>/runs/history/<workflow>/<run_id>.json`。若上一轮异常停在 `running`，归档前
会统一标记为 `stale`。同一项目的组合 workflow 还会使用 `runs/locks/<workflow>.lock`
避免并发覆盖。`status --json` 的 `runs` 字段列出各 workflow 当前状态和阶段详情。

按一次运行查看阶段产物、manifest、历史目录和 API 用量:

```bash
ai-clip run-status --workflow source-draft -p demo
ai-clip run-status --workflow source-draft -p demo --json
ai-clip run-status --workflow daily-radar --date 2026-07-09 --json
```

所有支持 `--json` 的命令统一返回:

```json
{"schema_version": 1, "command": "run-status", "status": "succeeded", "result": {}}
```

## 本地 smoke

Windows 本地快速 smoke test 不触发付费 API:

```powershell
.\scripts\smoke.ps1
```

它会运行 `doctor`、解析 `status --json`，并跑一组 targeted pytest。
