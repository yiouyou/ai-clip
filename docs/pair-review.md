# Pair Review

`pair-review` 把 ai-pair 的互审思路做成 Python 阶段：同一份文本产物交给两个不同模型审查，
避免同模型互审。

## 配置

`.env` 可配置 OpenAI-compatible review endpoint:

```text
NEWAPP_URL=https://.../v1
NEWAPP_API_KEY=...
DEEPSEEK_API_KEY=...
```

默认模型池包含:

```text
gpt-5.5
claude-sonnet-4-6
claude-opus-4.8
deepseek-4-pro
```

若某个模型失败，会尝试下一个可用且不同的模型。

## 审阅

```bash
ai-clip pair-review -p demo --artifact storyboard
ai-clip pair-review -p demo --artifact script
ai-clip pair-review -p demo --artifact source_draft
ai-clip pair-review -p demo --artifact research
ai-clip pair-review --artifact zack_draft --date 2026-07-07
```

支持产物:

```text
analysis
research
script
storyboard
source_draft
zack_draft
```

项目产物结果写入:

```text
data/<project>/reviews/<artifact>_review.json
```

Radar draft 结果写入:

```text
data/radar/reviews/YYYY-MM-DD_zack_draft_review.json
```

## 改写

当前 `--rewrite` 支持:

```bash
ai-clip pair-review -p demo --artifact research --rewrite
ai-clip pair-review -p demo --artifact script --rewrite
ai-clip pair-review -p demo --artifact source_draft --rewrite
ai-clip pair-review --artifact zack_draft --date 2026-07-07 --rewrite
```

输出:

```text
data/<project>/research.revised.md
data/<project>/script.revised.md
data/<project>/source_draft.revised.md
data/radar/drafts/YYYY-MM-DD.revised.md
```

`--rewrite` 是有界质量闭环：执行一次 review、一次 rewrite、一次 verify，然后停止；verify
即使仍给出 `revise` 或 `block`，也不会自动触发第二次改写。

验证报告:

```text
data/<project>/reviews/<artifact>_verify.json
data/radar/reviews/YYYY-MM-DD_zack_draft_verify.json
```

review、revised draft 和 verify report 都有 `<artifact>.meta.json`，会跟踪原稿、review report、
模型和 reviewer pool。`ai-clip status -p P` 与 `radar-status` 会显示三段 freshness。

机器可读输出:

```bash
ai-clip pair-review -p demo --artifact source_draft --rewrite --json
```
