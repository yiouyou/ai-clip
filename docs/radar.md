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

## 主流程

```bash
uv run --extra download --extra extract ai-clip daily-radar --top 3 --channel-timeout 60 --channel-workers 4
```

流程:

```text
collect -> zack-ranking -> source-content -> zack-selection -> [source-research] -> zack-draft
```

`source-research` 默认不自动跑；需要事实补充时:

```bash
ai-clip daily-radar --top 3 --research --research-searches 1
```

可选互审/改写:

```bash
ai-clip daily-radar --top 3 --research --review
ai-clip daily-radar --top 3 --research --rewrite
```

## 阶段命令

```bash
ai-clip collect --workflow daily-radar --date 2026-07-08 --force-collect
ai-clip zack-ranking --workflow daily-radar --date 2026-07-08 --top 3
ai-clip source-content --workflow daily-radar --date 2026-07-08
ai-clip zack-selection --workflow daily-radar --date 2026-07-08
ai-clip source-research --workflow daily-radar --date 2026-07-08 --max-searches 2
ai-clip zack-draft --workflow daily-radar --date 2026-07-08
```

## 运维命令

```bash
ai-clip doctor
ai-clip radar-status --date 2026-07-08
ai-clip radar-repair --date 2026-07-08
ai-clip radar-repair --date 2026-07-08 --apply
```

`doctor` 默认只做本地检查，不触发付费 LLM/Tavily 调用。`radar-repair` 很保守，只清理明确
无效的空 snapshot / candidates。

`radar-status` 还会显示关键 radar 产物的 freshness:

```text
candidates
selection
source_research
zack_draft
```

状态含义与项目级 `status` 一致：`fresh`、`stale`、`missing`。

## 产物

```text
data/radar/snapshots/YYYY-MM-DD.jsonl
data/radar/collect-reports/YYYY-MM-DD.json
data/radar/runs/YYYY-MM-DD.json
data/radar/candidates/YYYY-MM-DD.json
data/radar/source-content/YYYY-MM-DD/
data/radar/selections/YYYY-MM-DD.json
data/radar/selections/YYYY-MM-DD.md
data/radar/research/YYYY-MM-DD.json
data/radar/research/YYYY-MM-DD.md
data/radar/briefs/YYYY-MM-DD.md
data/radar/drafts/YYYY-MM-DD.md
data/radar/drafts/YYYY-MM-DD.revised.md
data/radar/reviews/YYYY-MM-DD_zack_draft_review.json
```

关键产物会额外写 sidecar metadata:

```text
data/radar/candidates/YYYY-MM-DD.json.meta.json
data/radar/selections/YYYY-MM-DD.json.meta.json
data/radar/research/YYYY-MM-DD.md.meta.json
data/radar/drafts/YYYY-MM-DD.md.meta.json
```

旧版 `scout:` 配置、`AICLIP_SCOUT_*` 环境变量和 `data/scout/` 历史产物会作为兼容输入读取；
新运行统一写入 `radar` 命名。

## 定时运行

建议用系统定时器每天 7 点跑一次:

```powershell
schtasks /Create /SC DAILY /ST 07:00 /TN "ai-clip daily-radar" /TR "powershell -NoProfile -Command cd E:\_Ai\my\ai-clip; uv run --extra download --extra extract ai-clip daily-radar --top 3 --channel-timeout 60"
```
