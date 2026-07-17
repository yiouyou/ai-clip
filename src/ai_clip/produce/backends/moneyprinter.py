"""MoneyPrinterTurbo backend: theme -> stock-footage + TTS + subtitle video via
its REST API (https://github.com/harry0703/MoneyPrinterTurbo).

Flow: POST /api/v1/videos -> task_id; poll GET /api/v1/tasks/{id} until the task
reports `videos`; download the first result to the requested path.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from ai_clip.core.async_jobs import (
    ACTIVE_JOB_STATUSES,
    AsyncJobState,
    async_job_request_hash,
    async_job_state_path,
    new_async_job_state,
    read_async_job_state,
    transition_async_job,
    write_async_job_state,
)
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
        out = Path(spec.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        body = self._request_body(spec)
        request_hash = async_job_request_hash(self.name, self.base_url, body)
        state = self._resumable_state(out, request_hash)
        if state is not None and state.status == "succeeded" and out.exists():
            return out

        if state is None:
            state = new_async_job_state(
                provider=self.name,
                request_hash=request_hash,
                output_path=out,
                status="submitting",
            )
            write_async_job_state(out, state)
            state = self._submit(body, out, state)

        video_url = self._await_video(state.remote_id, out, state)
        self._download(video_url, out)
        transition_async_job(out, state, "succeeded")
        return out

    def _resumable_state(self, out: Path, request_hash: str) -> AsyncJobState | None:
        try:
            state = read_async_job_state(out)
        except (OSError, ValueError) as exc:
            raise MoneyPrinterError(
                f"invalid async job state {async_job_state_path(out)}: {type(exc).__name__}"
            ) from exc
        if state is None:
            return None
        if state.provider != self.name or state.request_hash != request_hash:
            if state.status in ACTIVE_JOB_STATUSES:
                raise MoneyPrinterError(
                    f"active async job conflicts with this request: {async_job_state_path(out)}"
                )
            return None
        if state.status == "failed":
            return None
        if not state.remote_id:
            raise MoneyPrinterError(
                "MoneyPrinter submission outcome is unknown; verify the server before removing "
                f"{async_job_state_path(out)}"
            )
        return state

    def _submit(self, body: dict, out: Path, state: AsyncJobState) -> AsyncJobState:
        try:
            resp = httpx.post(
                f"{self.base_url}/api/v1/videos",
                json=body,
                timeout=60.0,
            )
            resp.raise_for_status()
            task_id = str(resp.json()["data"]["task_id"])
            if not task_id:
                raise ValueError("empty MoneyPrinter task id")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            transition_async_job(out, state, "failed", error=type(exc).__name__)
            raise MoneyPrinterError(
                "could not connect to MoneyPrinter; task was not submitted"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            terminal = status < 500
            transition_async_job(
                out,
                state,
                "failed" if terminal else "unknown",
                error=f"HTTP {status}",
            )
            message = "rejected" if terminal else "returned an ambiguous failure for"
            raise MoneyPrinterError(f"MoneyPrinter {message} submission with HTTP {status}") from exc
        except (httpx.RequestError, KeyError, TypeError, ValueError) as exc:
            transition_async_job(out, state, "unknown", error=type(exc).__name__)
            raise MoneyPrinterError(
                "MoneyPrinter submission outcome is unknown; inspect the job state before retrying"
            ) from exc
        return transition_async_job(out, state, "submitted", remote_id=task_id)

    @staticmethod
    def _download(video_url: str, out: Path) -> None:
        tmp = out.with_name(f".{out.name}.download.tmp")
        try:
            with httpx.stream("GET", video_url, timeout=120.0) as response:
                response.raise_for_status()
                with tmp.open("wb") as file:
                    for chunk in response.iter_bytes():
                        file.write(chunk)
            tmp.replace(out)
        finally:
            tmp.unlink(missing_ok=True)

    def _await_video(self, task_id: str, out: Path, state: AsyncJobState) -> str:
        deadline = time.monotonic() + self.timeout
        url = f"{self.base_url}/api/v1/tasks/{task_id}"
        marked_running = state.status == "running"
        while time.monotonic() < deadline:
            response = httpx.get(url, timeout=30.0)
            response.raise_for_status()
            data = response.json().get("data", {})
            if not marked_running:
                state = transition_async_job(out, state, "running")
                marked_running = True
            videos = data.get("videos") or data.get("combined_videos")
            if videos:
                video_url = videos[0]
                return (
                    video_url
                    if video_url.startswith("http")
                    else f"{self.base_url}{video_url}"
                )
            if data.get("state", 1) < 0:
                transition_async_job(out, state, "failed", error="remote task failed")
                raise MoneyPrinterError(f"task {task_id} failed: {data}")
            time.sleep(5.0)
        raise MoneyPrinterError(f"timed out waiting for MoneyPrinterTurbo task {task_id}")
