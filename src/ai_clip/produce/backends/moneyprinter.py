"""MoneyPrinterTurbo backend: theme -> stock-footage + TTS + subtitle video via
its REST API (https://github.com/harry0703/MoneyPrinterTurbo).

Flow: POST /api/v1/videos -> task_id; poll GET /api/v1/tasks/{id} until the task
reports `videos`; download the first result to the requested path.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from ai_clip.produce.backends.base import ProduceSpec

_VALID_ASPECT = {"9:16", "16:9", "1:1"}
_DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural-Female"  # MPT edge-tts voice id


class MoneyPrinterError(RuntimeError):
    pass


class MoneyPrinterBackend:
    name = "moneyprinter"

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def is_available(base_url: str, timeout: float = 3.0) -> bool:
        try:
            r = httpx.get(f"{base_url.rstrip('/')}/docs", timeout=timeout)
            return r.status_code < 500
        except Exception:
            return False

    def _request_body(self, spec: ProduceSpec) -> dict:
        aspect = spec.aspect_ratio if spec.aspect_ratio in _VALID_ASPECT else "9:16"
        body = {
            "video_subject": spec.theme,
            "video_aspect": aspect,
            "subtitle_enabled": spec.subtitle,
            "paragraph_number": spec.paragraphs,
            "voice_name": spec.voice_name or _DEFAULT_VOICE,
        }
        if spec.language:
            body["video_language"] = spec.language
        return body

    def produce(self, spec: ProduceSpec) -> Path:
        resp = httpx.post(
            f"{self.base_url}/api/v1/videos",
            json=self._request_body(spec),
            timeout=60.0,
        )
        resp.raise_for_status()
        task_id = resp.json()["data"]["task_id"]
        video_url = self._await_video(task_id)

        out = Path(spec.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", video_url, timeout=120.0) as r:
            r.raise_for_status()
            with out.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return out

    def _await_video(self, task_id: str) -> str:
        deadline = time.monotonic() + self.timeout
        url = f"{self.base_url}/api/v1/tasks/{task_id}"
        while time.monotonic() < deadline:
            data = httpx.get(url, timeout=30.0).json().get("data", {})
            videos = data.get("videos") or data.get("combined_videos")
            if videos:
                url = videos[0]
                return url if url.startswith("http") else f"{self.base_url}{url}"
            if data.get("state", 1) < 0:
                raise MoneyPrinterError(f"task {task_id} failed: {data}")
            time.sleep(5.0)
        raise MoneyPrinterError(f"timed out waiting for MoneyPrinterTurbo task {task_id}")
