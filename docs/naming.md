# Naming conventions

ai-clip separates names by responsibility. A workflow describes an outcome, a stage describes an
action, and an artifact describes persisted data. Personal editorial preferences belong in
configuration rather than Python packages, domain models, or stage identifiers.

## Syntax

| Surface | Convention | Example |
|---|---|---|
| CLI commands, workflows, stages | lowercase kebab-case | `daily-radar`, `rank-candidates` |
| Python modules, functions, fields | snake_case | `rank_candidates`, `editorial_profile` |
| Python types | PascalCase | `TopicSelection`, `EditorialDraft` |
| Artifact identifiers | lowercase kebab-case | `source-draft`, `editorial-draft` |
| Environment variables | uppercase snake case | `AICLIP_WEB_RESEARCH_MAX_SEARCHES` |

Chinese and English documentation may localize descriptions, but code identifiers remain English.
Product or person names may appear only as configured profile values.

## Canonical vocabulary

The public workflow names remain:

`transcribe`, `teardown`, `source-draft`, `remix`, `original`, and `daily-radar`.

Daily Radar uses the following action-oriented stage vocabulary:

```text
collect
-> rank-candidates
-> enrich-content
-> rerank-candidates
-> select-topic
-> research-evidence
-> draft-script
-> review-artifact
-> rewrite-artifact
-> verify-artifact
```

The corresponding artifact vocabulary is noun-oriented:

```text
snapshots -> shortlist -> source-content -> candidates -> selection
-> evidence-research -> editorial-draft -> review -> revised-draft -> verification
```

## Research boundaries

- `source-research`: facts and context grounded in a source transcript and its analysis.
- `topic-research`: research that starts from an original theme without a source transcript.
- `research-evidence`: stage that verifies claims and gathers evidence for a selected Radar topic.
- `evidence-research`: persisted report produced by `research-evidence`.

## Review boundaries

- `storyboard-edit`: human round trip between `storyboard.json` and the editable `script.md`.
- `review-artifact`: model-based quality review of a typed text artifact.
- `rewrite-artifact`: one bounded revision using an artifact review report.
- `verify-artifact`: review of the revised artifact without another rewrite.

`pair` is a review strategy, not the domain name. The review service may use two or more distinct
models, while the stage and artifact names remain independent of that implementation.

## Migration map

| Legacy name | Canonical name |
|---|---|
| project `research` command | `source-research` |
| `zack-ranking` | `rank-candidates` |
| `source-content` stage | `enrich-content` |
| `content-rerank` | `rerank-candidates` |
| `zack-selection` | `select-topic` |
| Radar `source-research` | `research-evidence` |
| `zack-draft` | `draft-script` stage / `editorial-draft` artifact |
| human `review` command | `storyboard-edit` |
| `pair-review` | `review-artifact` |
| `pair-rewrite` | `rewrite-artifact` |
| `pair-verify` | `verify-artifact` |
| `radar-status` | `daily-radar-diagnostics` |
| `radar-repair` | `daily-radar-repair` |
| `radar-feedback` | `daily-radar-feedback` |
| `ZackSelection` | `TopicSelection` |
| `ZackDraft` | `EditorialDraft` |
| `pair` config section | `artifact_review` |
| `source_research` config section | `web_research` |
| `AICLIP_SOURCE_RESEARCH_MAX_SEARCHES` | `AICLIP_WEB_RESEARCH_MAX_SEARCHES` |
| `AICLIP_SOURCE_RESEARCH_MAX_RESULTS` | `AICLIP_WEB_RESEARCH_MAX_RESULTS` |

See [the 0.2 upgrade guide](upgrade-0.2.md) for Python imports, agent tools, artifact filenames,
and the required upgrade sequence.

## Compatibility policy

Public command and configuration renames are coordinated changes, not permanent aliases. Legacy data
is never deleted or renamed automatically. `ai-clip migrate-names` previews the explicit, idempotent
migration; `--apply` performs it and refuses to overwrite an existing destination. Migration changes
only structured identifiers and artifact paths; reviewer output, errors, reasons, and other free-form
text remain unchanged.
