from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from ai_clip.core.artifacts import ArtifactStore
from ai_clip.core.artifacts import write_model as _write_model
from ai_clip.core.artifacts import write_text_atomic
from ai_clip.radar.models import RadarSnapshot, RadarVideo


class RadarPaths:
    def __init__(self, data_dir: str | Path, date: str) -> None:
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "radar"
        self.date = date

    @property
    def store(self) -> ArtifactStore:
        return ArtifactStore(self.root)

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def shortlists_dir(self) -> Path:
        return self.root / "shortlists"

    @property
    def briefs_dir(self) -> Path:
        return self.root / "briefs"

    @property
    def selections_dir(self) -> Path:
        return self.root / "selections"

    @property
    def research_dir(self) -> Path:
        return self.root / "research"

    @property
    def drafts_dir(self) -> Path:
        return self.root / "drafts"

    @property
    def reviews_dir(self) -> Path:
        return self.root / "reviews"

    @property
    def feedback_dir(self) -> Path:
        return self.root / "feedback"

    @property
    def source_content_dir(self) -> Path:
        return self.root / "source-content" / self.date

    @property
    def backfills_dir(self) -> Path:
        return self.root / "backfills"

    @property
    def collect_reports_dir(self) -> Path:
        return self.root / "collect-reports"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def backfill_run_dir(self, end_date: str) -> Path:
        return self.backfills_dir / end_date

    @property
    def snapshot_jsonl(self) -> Path:
        return self.snapshots_dir / f"{self.date}.jsonl"

    @property
    def candidates_json(self) -> Path:
        return self.candidates_dir / f"{self.date}.json"

    @property
    def shortlist_json(self) -> Path:
        return self.shortlists_dir / f"{self.date}.json"

    @property
    def feedback_events_jsonl(self) -> Path:
        return self.feedback_dir / "events.jsonl"

    @property
    def brief_md(self) -> Path:
        return self.briefs_dir / f"{self.date}.md"

    @property
    def selection_json(self) -> Path:
        return self.selections_dir / f"{self.date}.json"

    @property
    def selection_md(self) -> Path:
        return self.selections_dir / f"{self.date}.md"

    @property
    def research_json(self) -> Path:
        return self.research_dir / f"{self.date}.json"

    @property
    def research_md(self) -> Path:
        return self.research_dir / f"{self.date}.md"

    @property
    def draft_md(self) -> Path:
        return self.drafts_dir / f"{self.date}.md"

    @property
    def draft_revised_md(self) -> Path:
        return self.drafts_dir / f"{self.date}.revised.md"

    @property
    def run_status_json(self) -> Path:
        return self.runs_dir / f"{self.date}.json"

    @property
    def collect_report_json(self) -> Path:
        return self.collect_reports_dir / f"{self.date}.json"

    def ensure(self) -> None:
        for path in (
            self.snapshots_dir,
            self.candidates_dir,
            self.shortlists_dir,
            self.briefs_dir,
            self.selections_dir,
            self.research_dir,
            self.drafts_dir,
            self.reviews_dir,
            self.feedback_dir,
            self.source_content_dir,
            self.backfills_dir,
            self.collect_reports_dir,
            self.runs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def write_json_model(path: Path, model: BaseModel) -> None:
    _write_model(path, model)


def append_snapshots(path: Path, snapshots: list[RadarSnapshot]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for snapshot in snapshots:
            f.write(snapshot.model_dump_json() + "\n")


def write_snapshots(path: Path, snapshots: list[RadarSnapshot]) -> None:
    content = "".join(snapshot.model_dump_json() + "\n" for snapshot in snapshots)
    write_text_atomic(path, content, encoding="utf-8")


def read_snapshots(path: Path) -> list[RadarSnapshot]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(RadarSnapshot.model_validate(json.loads(line)))
    return out


def dedupe_snapshots(snapshots: list[RadarSnapshot]) -> list[RadarSnapshot]:
    by_video_id: dict[str, RadarSnapshot] = {}
    for snapshot in snapshots:
        by_video_id[snapshot.video.video_id] = snapshot
    return list(by_video_id.values())


def latest_previous_by_video(root: Path, before_date: str) -> dict[str, RadarVideo]:
    latest: dict[str, tuple[datetime, RadarVideo]] = {}
    latest.update(_latest_previous_by_video(root, before_date, latest))
    return {video_id: item[1] for video_id, item in latest.items()}


def _latest_previous_by_video(
    root: Path,
    before_date: str,
    latest: dict[str, tuple[datetime, RadarVideo]],
) -> dict[str, tuple[datetime, RadarVideo]]:
    snapshots_dir = root / "snapshots"
    if not snapshots_dir.exists():
        return latest
    for path in sorted(snapshots_dir.glob("*.jsonl")):
        if path.stem >= before_date:
            continue
        for snapshot in read_snapshots(path):
            try:
                ts = datetime.fromisoformat(snapshot.collected_at)
            except ValueError:
                ts = datetime.min
            current = latest.get(snapshot.video.video_id)
            if current is None or ts > current[0]:
                latest[snapshot.video.video_id] = (ts, snapshot.video)
    return latest


