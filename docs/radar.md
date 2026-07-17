# Daily Radar 选题流程

`daily-radar` 用于每天追踪指定 YouTube/B站频道，收集 metadata，选出 Top 3，再选定一个
主选题生成原创口播草稿。

## 频道配置

频道列表手写在 `config/channels.yaml`，可从 `config/channels.example.yaml` 复制:

```yaml
channels:
  - platform: youtube
    url: https://www.youtube.com/@channel
    name: channel-name
    pool: ai
    role: signal
    tags: [ai, business]
    priority: 1
    lens_fit: 1.3
    max_duration_sec: 1800
    cookies: ""

  - platform: bilibili
    url: https://space.bilibili.com/123456
    name: up-name
    pool: society
    role: signal
    tags: [tech, china]
    priority: 1
    lens_fit: 1.4
    max_duration_sec: 1800
    cookies: path/to/bilibili.cookies.txt
```

`cookies` 可为空；需要登录时填 Netscape cookies txt。`pool` 用于主题分组，`role` 可为
`signal`、`reference` 或 `style`，`lens_fit` 表示与“用生物学/复杂系统解释政治经济科技”
偏好的贴合度。

B站频道使用两阶段采集：先读取最多 `radar.channel_limit` 个按时间倒序的 BV 号，再展开最多
`radar.bilibili_detail_limit` 个详情（默认 8）。发现窗口内视频后遇到首条过期视频会提前停止；
单条详情出现 HTTP 412 等错误时会跳过并把频道标记为 `partial`，已经获得的视频仍会进入快照。
这两个上限和 `radar.since_days` 均可在 `config/default.yaml` 或自定义配置中调整。
回顾性 backfill 会把 B站详情展开上限提升到本次 `channel_limit`，不会沿用每日默认的 8 条。

## 主流程

```bash
uv run --extra download --extra extract ai-clip daily-radar --top 3 --channel-timeout 60 --channel-workers 4
```

流程:

```text
collect -> zack-ranking -> source-content -> content-rerank -> zack-selection
        -> [source-research] -> zack-draft
```

完整流程顺序和 optional 条件来自统一 `WorkflowSpec`。阶段实现保留在 `radar/stage.py`，完整
编排在 `radar/workflow.py`，回顾性回填在 `radar/backfill.py`，超时采集在 `radar/collect.py`。

`zack-ranking` 先按频道/平台基线归一化 metadata，默认保留 9 个 shortlist；`source-content`
只为 shortlist 获取字幕或转写，`content-rerank` 再结合脚本深度、机制素材和 metadata 排名收敛
到最终 Top 3。这样不会全量转写几十个频道，也不会在拿到脚本前过早淘汰内容更合适的选题。

每个 `script.json` 都有版本化 manifest，记录 URL、内容来源、Whisper 设置和 cookie 文件签名。
更新 cookie 后会先重试官方字幕；字幕仍不可用且 Whisper 配置未变时复用原转写，模型或语言改变时
才重新转写。旧字幕缓存会直接补 manifest，旧 Whisper 缓存会先尝试一次字幕升级。

`source-research` 默认由选题的 `fact_risk` 自动决定：高风险且 Tavily 可用时最多搜索 2 次；
手动 `--research` 会强制执行，仍可用 `--research-searches` 指定 1-3 次:

```bash
ai-clip daily-radar --top 3 --research --research-searches 1
```

可选互审/改写:

```bash
ai-clip daily-radar --top 3 --research --review
ai-clip daily-radar --top 3 --research --rewrite
```

`--rewrite` 固定执行一次 pair-review、一次 rewrite 和一次 verify，不会循环改写。

## 阶段命令

```bash
ai-clip collect --workflow daily-radar --date 2026-07-08 --force-collect
ai-clip zack-ranking --workflow daily-radar --date 2026-07-08 --top 3
ai-clip source-content --workflow daily-radar --date 2026-07-08
ai-clip content-rerank --workflow daily-radar --date 2026-07-08
ai-clip zack-selection --workflow daily-radar --date 2026-07-08
ai-clip source-research --workflow daily-radar --date 2026-07-08 --max-searches 2
ai-clip zack-draft --workflow daily-radar --date 2026-07-08
```

强制采集按频道更新：成功频道替换旧快照；`partial`、`failed` 或 `timeout` 频道保留已有快照，
并合并本次成功获取的视频，避免临时网络或 cookie 问题清空当天数据。

## 运维命令

```bash
ai-clip doctor
ai-clip radar-status --date 2026-07-08
ai-clip radar-status --date 2026-07-08 --json
ai-clip run-status --workflow daily-radar --date 2026-07-08 --json
ai-clip radar-repair --date 2026-07-08
ai-clip radar-repair --date 2026-07-08 --apply
ai-clip radar-feedback accept --date 2026-07-08 --reason "角度适合"
ai-clip radar-feedback reject --date 2026-07-08 --video-id youtube:VIDEO_ID --reason "过于气象"
```

`doctor` 默认只做本地检查，不触发付费 LLM/Tavily 调用。`radar-repair` 很保守，只清理明确
无效的空 snapshot / shortlist / candidates。

`radar-status` 还会显示关键 radar 产物的 freshness:

```text
shortlist
candidates
selection
source_research
zack_draft
pair_review
pair_rewrite
pair_verify
```

状态含义与项目级 `status` 一致：`fresh`、`stale`、`missing`。
`--json` 使用统一 envelope，并输出 `run_id`、attempt、阶段状态、频道诊断、产物 freshness
以及本次 LLM/TTS/Tavily calls、tokens 和 cost。`run-status` 还会按阶段列出产物路径、manifest
和历史运行目录。
每次完整 `daily-radar` 重跑会把上一轮状态归档到
`data/radar/runs/history/YYYY-MM-DD/<run_id>.json`；日期锁与项目 workflow 使用同一套安全
PID 检测，不会在 Windows 上通过 `os.kill(pid, 0)` 探测进程。

## 产物

```text
data/radar/snapshots/YYYY-MM-DD.jsonl
data/radar/collect-reports/YYYY-MM-DD.json
data/radar/runs/YYYY-MM-DD.json
data/radar/shortlists/YYYY-MM-DD.json
data/radar/candidates/YYYY-MM-DD.json
data/radar/feedback/events.jsonl
data/radar/source-content/YYYY-MM-DD/
data/radar/selections/YYYY-MM-DD.json
data/radar/selections/YYYY-MM-DD.md
data/radar/research/YYYY-MM-DD.json
data/radar/research/YYYY-MM-DD.md
data/radar/briefs/YYYY-MM-DD.md
data/radar/drafts/YYYY-MM-DD.md
data/radar/drafts/YYYY-MM-DD.revised.md
data/radar/reviews/YYYY-MM-DD_zack_draft_review.json
data/radar/reviews/YYYY-MM-DD_zack_draft_verify.json
```

关键产物会额外写 sidecar metadata:

```text
data/radar/shortlists/YYYY-MM-DD.json.meta.json
data/radar/candidates/YYYY-MM-DD.json.meta.json
data/radar/selections/YYYY-MM-DD.json.meta.json
data/radar/research/YYYY-MM-DD.md.meta.json
data/radar/drafts/YYYY-MM-DD.md.meta.json
```

## 定时运行

建议用系统定时器每天 7 点跑一次:

```powershell
schtasks /Create /SC DAILY /ST 07:00 /TN "ai-clip daily-radar" /TR "powershell -NoProfile -Command cd E:\_Ai\my\ai-clip; uv run --extra download --extra extract ai-clip daily-radar --top 3 --channel-timeout 60"
```
