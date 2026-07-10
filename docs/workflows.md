# 工作流说明

ai-clip 的普通项目流程围绕 `data/<project>/` 组织。每个阶段都可以单独跑，也可以通过组合
workflow 串起来。

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

JSON 输出包含 `research/storyboard/source_draft` 的 `fresh/stale/missing`、分镜镜头数、
缺失素材和 `assets_ready`。

## 组合 workflow

| 命令 | 流程 | 产出 |
|------|------|------|
| `ai-clip transcribe <url> -p P` | download -> extract -> export | `.srt` + `.txt` |
| `ai-clip teardown <url> -p P` | download -> extract -> analyze | 爆款公式 |
| `ai-clip source-draft <url> -p P` | download -> extract -> analyze -> [research] -> source_draft | `source_draft.md` |
| `ai-clip remix <url> --theme T -p P` | download -> extract -> analyze -> [research] -> remix storyboard -> voiceover -> assemble | `output.mp4` |
| `ai-clip original --theme T -p P` | [research] -> storyboard -> assets -> voiceover -> assemble | `output.mp4` 或提示补素材 |

`source-draft`、`remix`、`original` 都可显式启用项目级 research:

```bash
ai-clip source-draft "<url>" -p demo --research --theme "..." --research-searches 1
ai-clip remix "<url>" -p demo --theme "..." --research --research-searches 1
ai-clip original -p demo --theme "..." --research --research-searches 1
```

`source-draft` 默认复用已有中间产物: `clip.json`、`transcript.json`、`analysis.json`、
`research.md` 和 `source_draft.md`。需要强制重跑时使用 `--no-resume`。长视频在 CPU
上转写较慢时,可用 `--whisper-model small` 临时降低本次转写模型。

组合 workflow 会写运行状态:

```text
data/<project>/runs/<workflow>.json
```

## 本地 smoke

Windows 本地快速 smoke test 不触发付费 API:

```powershell
.\scripts\smoke.ps1
```

它会运行 `doctor`、解析 `status --json`，并跑一组 targeted pytest。
