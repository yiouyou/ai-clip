# Research 阶段

Research 分为两种输入契约，共享搜索引擎和产物格式：

- `research`：源视频研究，依赖 transcript，可选读取 analysis，供 source-draft/remix 使用。
- `topic-research`：主题研究，只依赖 theme，供 original 内部使用。

两者都用于在生成口播/分镜前补足事实、背景、风险边界和原创角度。

## 命令

```bash
ai-clip research -p demo --theme "AI 公司的生态位竞争" --max-searches 2
```

`--max-searches` 会被限制在 1-3 次。默认复用 `.env` 中的 `TAVILY_API_KEY` 和当前 LLM 配置。
每个 query 遇到 429、5xx 或临时网络错误时最多按 `source_research.max_attempts` 尝试 2 次；
这不增加检索角度数量。认证错误、其他 4xx 和非法响应不会重试。

## 产物

```text
data/<project>/research.json
data/<project>/research.md
data/<project>/research.json.meta.json
data/<project>/research.md.meta.json
```

`research.md` 是可编辑文件。`storyboard` 和 `source-draft` 会读取它，因此可以人工改完
研究笔记再继续生成。

## 注入位置

- `storyboard` 自动读取 `research.md`
- `source-draft` 自动读取 `research.md`
- `remix --research` 会在分镜前先跑 research
- `original --research` 会在分镜前运行不依赖 transcript 的 `topic-research`

## Freshness

关键产物会写 `<artifact>.meta.json`，记录生成阶段、输入文件签名、参数和模型。

```bash
ai-clip status -p demo
ai-clip status -p demo --json
```

`fresh` 表示输入未变，`stale` 表示输入签名变化或 manifest 缺失，`missing` 表示产物不存在。
