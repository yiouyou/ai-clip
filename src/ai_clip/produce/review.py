"""Human review of the script (文案) between storyboard and voiceover/assemble.

The storyboard.json is the contract assemble reads, but hand-editing JSON (with
Chinese narration) is awkward. So we round-trip through a friendly script.md:
machine exports it, the human edits narration (and, for remix, the [start-end]
timestamps, or deletes a block to drop a shot), machine parses it back into the
storyboard. Shots keep their original index/asset filenames; removed blocks drop.
"""

from __future__ import annotations

import re

from ai_clip.core.models import Storyboard, VideoFormat

_HEADER = re.compile(r"^##\s*shot\s*(\d+)\s*(?:\[([0-9.]+)\s*-\s*([0-9.]+)\])?", re.I)


def to_script_md(sb: Storyboard) -> str:
    lines = [
        f"# Script: {sb.project}  (format: {sb.format})",
        "# Edit the narration under each shot, then: ai-clip review -p "
        f"{sb.project} --apply",
    ]
    if sb.format == VideoFormat.remix:
        lines.append("# Remix: you may also adjust the [start-end] timestamps (seconds).")
    lines.append("")
    for shot in sb.shots:
        if shot.is_source_segment:
            head = f"## shot {shot.index:02d}  [{shot.source_start:g}-{shot.source_end:g}]"
        else:
            head = f"## shot {shot.index:02d}"
        lines += [head, shot.voiceover.strip(), ""]
    return "\n".join(lines)


def apply_script_md(sb: Storyboard, text: str, source_max: float | None = None) -> Storyboard:
    """Parse an edited script.md back into the storyboard. Returns a new
    Storyboard keeping original shots' non-narration fields."""
    by_index = {s.index: s for s in sb.shots}
    blocks: list[tuple[int, float | None, float | None, list[str]]] = []
    cur: tuple[int, float | None, float | None] | None = None
    buf: list[str] = []

    def flush():
        if cur is not None:
            blocks.append((cur[0], cur[1], cur[2], buf.copy()))

    for line in text.splitlines():
        if line.startswith("#") and not line.startswith("##"):
            continue  # comment / title
        m = _HEADER.match(line)
        if m:
            flush()
            buf.clear()
            idx = int(m.group(1))
            start = float(m.group(2)) if m.group(2) else None
            end = float(m.group(3)) if m.group(3) else None
            cur = (idx, start, end)
        elif cur is not None:
            buf.append(line)
    flush()

    new_shots = []
    for idx, start, end, body in blocks:
        base = by_index.get(idx)
        if base is None:
            continue
        shot = base.model_copy()
        narration = "\n".join(body).strip()
        if narration:
            shot.voiceover = narration
        if shot.is_source_segment and start is not None and end is not None:
            start = max(0.0, start)
            if source_max is not None:
                end = min(end, source_max)
            if end > start:
                shot.source_start = start
                shot.source_end = end
                shot.duration_sec = round(end - start, 3)
        new_shots.append(shot)

    return sb.model_copy(update={"shots": new_shots})
