"""NarratoAI backend (解说二创) by hard-wiring its internal pipeline.

NarratoAI ships no HTTP API (WebUI-only), so we invoke its core function
`app.services.task.start_subclip_unified(task_id, VideoClipParams)` in a
subprocess that runs inside NarratoAI's own repo + venv. We pass a pre-built
clip script (video_clip_json: spans + narration) plus the source video; the
runner cuts, narrates (TTS), and merges into a finished mp4.

Generating the clip script itself (vision analysis of the source) is left to
NarratoAI's WebUI or to ai-clip's own `remix` analyze→storyboard, which already
produces source spans + narration we can hand off here.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ai_clip.core.models import Storyboard

_RUNNER = Path(__file__).with_name("_narrato_runner.py")


class NarratoError(RuntimeError):
    pass


def storyboard_to_clip_json(sb: Storyboard) -> list[dict]:
    """Map a remix Storyboard (source spans + narration) to NarratoAI's clip
    script shape (timestamp range + narration per segment)."""
    out = []
    for shot in sb.shots:
        if not shot.is_source_segment:
            continue
        out.append({
            "timestamp": f"{shot.source_start:.1f}-{shot.source_end:.1f}",
            "picture": "",
            "narration": shot.voiceover,
            "OST": 0,
        })
    return out


class NarratoBackend:
    name = "narrato"

    def __init__(self, narrato_dir: str | Path, python_exe: str | Path):
        self.narrato_dir = Path(narrato_dir)
        self.python_exe = str(python_exe)

    def produce_remix(
        self,
        source_video: str | Path,
        clip_json: list[dict],
        out_path: str | Path,
        voice_name: str = "zh-CN-YunjianNeural",
        timeout: float = 1800.0,
    ) -> Path:
        if not self.narrato_dir.exists():
            raise NarratoError(f"NarratoAI repo not found: {self.narrato_dir}")
        out = Path(out_path)
        payload = {
            "video_origin_path": str(source_video),
            "video_clip_json": clip_json,
            "voice_name": voice_name,
            "out_path": str(out),
        }
        proc = subprocess.run(
            [self.python_exe, str(_RUNNER)],
            input=json.dumps(payload),
            cwd=str(self.narrato_dir),
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            raise NarratoError(f"NarratoAI runner failed:\n{proc.stderr[-2000:]}")
        result = proc.stdout.strip().splitlines()[-1]
        produced = Path(json.loads(result)["output"])
        if not produced.exists():
            raise NarratoError(f"NarratoAI reported success but no file at {produced}")
        return produced
