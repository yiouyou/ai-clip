"""Runner executed INSIDE NarratoAI's repo + venv (not ai-clip's).

Reads a JSON job from stdin, builds a VideoClipParams, runs NarratoAI's
start_subclip_unified, and prints {"output": "<path>"} as the last stdout line.
Kept dependency-free w.r.t. ai-clip — it only imports NarratoAI modules.
"""

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path


def main() -> None:
    # Running by absolute path doesn't put the NarratoAI repo (our cwd) on sys.path.
    sys.path.insert(0, os.getcwd())
    job = json.load(sys.stdin)

    from app.models.schema import VideoClipParams  # noqa: PLC0415
    from app.services import task as task_service  # noqa: PLC0415

    # start_subclip_unified reads the clip script from a file path, so dump it.
    script_path = Path(tempfile.gettempdir()) / f"aiclip_narrato_{uuid.uuid4().hex}.json"
    script_path.write_text(
        json.dumps(job["video_clip_json"], ensure_ascii=False), encoding="utf-8"
    )

    params = VideoClipParams(
        video_origin_path=job["video_origin_path"],
        video_clip_json=job["video_clip_json"],
        video_clip_json_path=str(script_path),
        voice_name=job.get("voice_name", "zh-CN-YunjianNeural"),
        subtitle_enabled=job.get("subtitle_enabled", True),
    )
    task_id = uuid.uuid4().hex
    result = task_service.start_subclip_unified(task_id, params)

    output = ""
    if isinstance(result, dict):
        videos = result.get("videos") or result.get("combined_videos") or []
        output = videos[0] if videos else result.get("output", "")
    print(json.dumps({"output": output or job.get("out_path", "")}))


if __name__ == "__main__":
    main()
