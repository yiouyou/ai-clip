# Upgrade to 0.2

Version 0.2 replaces implementation-specific and person-specific names with stable workflow,
stage, and artifact vocabulary. It is intentionally incompatible with the 0.1 names and does not
provide permanent aliases.

## Upgrade sequence

1. Back up the configured data directory and any custom configuration.
2. Update YAML, environment variables, shell scripts, Python imports, and tool calls using the
   tables below.
3. Run `ai-clip migrate-names` to preview persisted data changes.
4. Resolve every reported destination conflict, then run `ai-clip migrate-names --apply`.
5. Run `ai-clip doctor` and the workflow-specific smoke test used by your automation.

`migrate-names` changes only data artifacts, manifests, run metadata, and cost records. It never
edits YAML, `.env`, Python code, or external scripts.

## CLI commands

| 0.1 command | 0.2 command |
|---|---|
| `research` | `source-research` |
| Radar `source-research` | `research-evidence` |
| `zack-ranking` | `rank-candidates` |
| `source-content` | `enrich-content` |
| `content-rerank` | `rerank-candidates` |
| `zack-selection` | `select-topic` |
| `zack-draft` | `draft-script` |
| `review` | `storyboard-edit` |
| `pair-review` | `review-artifact` |
| `radar-status` | `daily-radar-diagnostics` |
| `radar-repair` | `daily-radar-repair` |
| `radar-feedback` | `daily-radar-feedback` |

The 0.1 Radar command named `source-research` and the 0.2 project command named
`source-research` are different operations. Radar automation must use `research-evidence`.

## Configuration

| 0.1 name | 0.2 name |
|---|---|
| YAML `pair` | YAML `artifact_review` |
| YAML `source_research` | YAML `web_research` |
| `AICLIP_SOURCE_RESEARCH_MAX_SEARCHES` | `AICLIP_WEB_RESEARCH_MAX_SEARCHES` |
| `AICLIP_SOURCE_RESEARCH_MAX_RESULTS` | `AICLIP_WEB_RESEARCH_MAX_RESULTS` |

Known 0.1 configuration names now raise an error that includes the replacement. API keys such as
`NEWAPP_API_KEY`, `NEWAPP_URL`, and `TAVILY_API_KEY` are unchanged.

## Python API

| 0.1 symbol | 0.2 symbol |
|---|---|
| `run_research` | `run_source_research` |
| `run_review_export` | `run_storyboard_edit_export` |
| `run_review_apply` | `run_storyboard_edit_apply` |
| `run_pair_review` | `run_artifact_review` |
| `run_pair_rewrite` | `run_artifact_rewrite` |
| `run_pair_verify` | `run_artifact_verify` |
| `run_zack_ranking` | `run_rank_candidates` |
| `run_source_content` | `run_enrich_content` |
| `run_content_rerank` | `run_rerank_candidates` |
| `run_zack_selection` | `run_select_topic` |
| Radar `run_source_research` | `run_research_evidence` |
| `run_zack_draft` | `run_draft_script` |
| `PairReviewReport` | `ArtifactReviewReport` |
| `ZackSelection` | `TopicSelection` |
| `ZackDraft` | `EditorialDraft` |

Imports under `ai_clip.pair`, `ai_clip.source_content`, `ai_clip.source_research`, and the
`ai_clip.zack_*` packages must move to `ai_clip.artifact_review`,
`ai_clip.content_enrichment`, `ai_clip.web_research`, or the corresponding `ai_clip.radar` module.

## Agent tools

| 0.1 tool | 0.2 tool |
|---|---|
| `research` | `source_research` |
| `pair_review` | `review_artifact` |
| `pair_rewrite` | `rewrite_artifact` |
| `pair_verify` | `verify_artifact` |
| `zack_ranking` | `rank_candidates` |
| `source_content` | `enrich_content` |
| `content_rerank` | `rerank_candidates` |
| `zack_selection` | `select_topic` |
| Radar `source_research` | `research_evidence` |
| `zack_draft` | `draft_script` |

## Persisted artifacts

The migration command covers the canonical filename and metadata changes, including:

```text
source_draft.md                 -> source-draft.md
source_draft.revised.md         -> source-draft.revised.md
voice_reference.mp3             -> voice-reference.mp3
source_draft_review.json        -> source-draft.review.json
source_draft_verify.json        -> source-draft.verify.json
*_zack_draft_review.json        -> *.editorial-draft.review.json
*_zack_draft_verify.json        -> *.editorial-draft.verify.json
```

The migration refuses to overwrite an existing destination. A conflict stops the apply operation so
the operator can compare both files before retrying.
