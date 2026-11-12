# Artifact Review

`review-artifact` is the typed multi-model review service. It sends one text artifact to two
different reviewer models, so the producer and reviewers do not collapse into one model perspective.
`pair` is the default strategy, not the public capability name.

## Configuration

`.env` configures OpenAI-compatible reviewer endpoints:

```text
NEWAPP_URL=https://.../v1
NEWAPP_API_KEY=...
DEEPSEEK_API_KEY=...
```

The default pool contains `gpt-5.5`, `claude-sonnet-4-6`, `claude-opus-4.8`, and
`deepseek-4-pro`. A failed reviewer falls through to the next available distinct model. YAML settings
live under `artifact_review`, while the main producer remains under `llm`.

## Review

```bash
ai-clip review-artifact -p demo --artifact storyboard
ai-clip review-artifact -p demo --artifact script
ai-clip review-artifact -p demo --artifact source-draft
ai-clip review-artifact -p demo --artifact research
ai-clip review-artifact --artifact editorial-draft --date 2026-07-07
```

Supported artifact identifiers are:

```text
analysis
research
script
storyboard
source-draft
editorial-draft
```

Project reports use `data/<project>/reviews/<artifact>.review.json`. Radar reports use
`data/radar/reviews/YYYY-MM-DD.editorial-draft.review.json`.

## Rewrite And Verify

```bash
ai-clip review-artifact -p demo --artifact research --rewrite
ai-clip review-artifact -p demo --artifact script --rewrite
ai-clip review-artifact -p demo --artifact source-draft --rewrite
ai-clip review-artifact --artifact editorial-draft --date 2026-07-07 --rewrite
```

The bounded chain runs `review-artifact -> rewrite-artifact -> verify-artifact` once and stops.
A verification verdict of `revise` or `block` never triggers a second automatic rewrite.

Revised files use `research.revised.md`, `script.revised.md`, `source-draft.revised.md`, or the Radar
path `drafts/YYYY-MM-DD.revised.md`. Verification reports use `<artifact>.verify.json` and
`YYYY-MM-DD.editorial-draft.verify.json` respectively.

Review, revised, and verification artifacts carry `.meta.json` sidecars that track source inputs,
review reports, producer model, and reviewer pool. `ai-clip status -p P` and
`ai-clip daily-radar-diagnostics` expose the three freshness states.

Machine-readable output uses the standard envelope:

```bash
ai-clip review-artifact -p demo --artifact source-draft --rewrite --json
```
