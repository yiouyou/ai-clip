# Changelog

## 0.2.0 - 2026-11-12

### Breaking changes

- Replaced person-specific Radar stage and Python names with domain names.
- Replaced `pair` review terminology with typed `artifact_review` terminology.
- Separated project `source-research`, theme `topic-research`, and Radar `research-evidence`.
- Renamed human storyboard editing from `review` to `storyboard-edit`.
- Renamed project draft and review artifact files to canonical kebab-case names.
- Renamed the `pair` and `source_research` YAML sections to `artifact_review` and
  `web_research`.
- Renamed the corresponding CLI commands, Python APIs, and agent tool identifiers.

No permanent compatibility aliases are provided. Follow [the 0.2 upgrade guide](docs/upgrade-0.2.md)
and run `ai-clip migrate-names` before `ai-clip migrate-names --apply` for persisted data.

### Added

- Explicit, idempotent migration for legacy artifact files, manifests, run metadata, and cost
  records.
- Targeted errors for known 0.1 YAML keys and environment variables.
- Public CLI, configuration, and tool contract tests.
