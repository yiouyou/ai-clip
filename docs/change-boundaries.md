# Change Boundaries

This file records the proposed commit boundaries for the current large worktree.
Do not treat it as a commit log; it is a staging guide for review.

## Do Not Commit By Default

- `config/channels.yaml` - personal channel list and local cookie file paths.
- `config/radar-fast.yaml` - local fast-run/debug config.
- `config/radar-feedback.yaml` - personal tuning file; commit only the example.
- `products/ask-china-why/` and `products/society-through-biology/` - personal content plans/scripts.
- `baseline.md` - temporary baseline note, currently superseded by docs/tests.
- `.env` and any `*cookies.txt` files are already ignored.

Current audit result:

- The files above are ignored by `.gitignore`.
- `products/mahjong.example.yaml` is an existing tracked example and is not affected by the ignore rule.
- Sensitive scan only found placeholders, docs examples, or test fixture strings:
  `NEWAPP_API_KEY=...`, `DEEPSEEK_API_KEY=...`, `<TAVILY_API_KEY>`, and `cookies.txt`.

## Commit Group 1: Radar Daily Workflow

Purpose: introduce the daily channel-tracking and topic-selection workflow.

Likely files:

- `src/ai_clip/radar/`
- `src/ai_clip/source_content/`
- `src/ai_clip/zack_ranking/`
- `src/ai_clip/zack_selection/`
- `src/ai_clip/zack_draft/`
- `config/channels.example.yaml`
- `config/radar-feedback.example.yaml`
- radar-related CLI entries in `src/ai_clip/cli.py`
- radar tests in `tests/test_radar.py` and CLI tests

## Commit Group 2: Source Research and Project Research

Purpose: add Tavily-backed research for selected radar topics and project workflows.

Likely files:

- `src/ai_clip/source_research/`
- `src/ai_clip/research/`
- project research wiring in `src/ai_clip/pipeline.py`
- prompt injection in `src/ai_clip/produce/formats/`
- research-related tests in `tests/test_source_research.py` and `tests/test_research.py`

## Commit Group 3: Source Draft and Pair Review

Purpose: support single-source original drafts and multi-model artifact review.

Likely files:

- `src/ai_clip/produce/source_draft.py`
- `src/ai_clip/pair/`
- source-draft and pair-review CLI entries
- `tests/test_source_draft.py`
- `tests/test_pair.py`

## Commit Group 4: ArtifactStore, Metadata, Status, and Run Status

Purpose: add atomic artifact primitives, freshness sidecars, and observable workflow status.

Likely files:

- `src/ai_clip/core/artifacts.py`
- `src/ai_clip/core/artifact_status.py`
- `src/ai_clip/core/run_status.py`
- `src/ai_clip/core/stages.py`
- metadata wiring in `src/ai_clip/pipeline.py` and `src/ai_clip/radar/stage.py`
- status CLI changes
- `tests/test_core.py`

## Commit Group 5: Assets Providers and Smart Illustrator

Purpose: route per-shot asset generation through configured engines.

Likely files:

- `src/ai_clip/produce/assets/factory.py`
- `src/ai_clip/produce/assets/smart_illustrator.py`
- `src/ai_clip/core/models.py`
- `src/ai_clip/produce/formats/*`
- `tests/test_produce.py`

## Commit Group 6: Docs, Smoke, and Operational Notes

Purpose: document workflows and add local non-paid checks.

Likely files:

- `README.md`
- `README-en.md`
- `docs/`
- `scripts/smoke.ps1`
- `AGENTS.md`
- `notes.md` if the team wants session/operations notes tracked

## Suggested Staging Commands

These commands are intentionally split by review boundary. Run only after reviewing
the corresponding diff.

```bash
# Group 1: radar workflow.
git add .gitignore config/default.yaml config/channels.example.yaml config/radar-feedback.example.yaml \
  src/ai_clip/radar src/ai_clip/source_content src/ai_clip/zack_ranking \
  src/ai_clip/zack_selection src/ai_clip/zack_draft tests/test_radar.py
git add -p src/ai_clip/cli.py src/ai_clip/workflows.py src/ai_clip/tools.py \
  tests/test_cli.py tests/test_tools.py tests/test_workflows.py

# Group 2: project/source research.
git add src/ai_clip/research src/ai_clip/source_research \
  tests/test_research.py tests/test_source_research.py
git add -p src/ai_clip/pipeline.py src/ai_clip/produce/formats

# Group 3: source draft and pair review.
git add src/ai_clip/produce/source_draft.py src/ai_clip/pair \
  tests/test_source_draft.py tests/test_pair.py
git add -p src/ai_clip/cli.py src/ai_clip/tools.py

# Group 4: artifact metadata and status.
git add src/ai_clip/core/artifacts.py src/ai_clip/core/artifact_status.py \
  src/ai_clip/core/run_status.py src/ai_clip/core/stages.py src/ai_clip/core/paths.py
git add -p src/ai_clip/pipeline.py src/ai_clip/radar/stage.py tests/test_core.py

# Group 5: asset provider routing and Smart Illustrator.
git add src/ai_clip/produce/assets/factory.py src/ai_clip/produce/assets/smart_illustrator.py
git add -p src/ai_clip/core/models.py src/ai_clip/produce/storyboard.py tests/test_produce.py

# Group 6: docs, smoke, and operations.
git add README.md README-en.md AGENTS.md docs scripts/smoke.ps1 .env.example pyproject.toml uv.lock
```

## Pre-Commit Checks

Run:

```bash
uv run ruff check src tests
uv run --extra download pytest -q
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/smoke.ps1
```
