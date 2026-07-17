"""ComfyUI API provider: submit a workflow JSON template with the shot's prompt
injected, poll for completion, and save the resulting image.

The workflow template is a ComfyUI "API format" graph exported from the UI. We
only swap the positive-prompt text into nodes tagged with AICLIP_PROMPT.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from pathlib import Path

import httpx

from ai_clip.core.artifacts import write_bytes_atomic
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
from ai_clip.core.models import Shot

_PROMPT_PLACEHOLDER = "AICLIP_PROMPT"


class ComfyUIError(RuntimeError):
    pass


class _ComfyUIJobFailed(ComfyUIError):
    pass


class ComfyUIProvider:
    name = "comfyui"

    def __init__(self, base_url: str, workflow_template: dict, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.workflow_template = workflow_template
        self.timeout = timeout

    @classmethod
    def from_file(cls, base_url: str, template_path: str | Path, **kw) -> "ComfyUIProvider":
        data = json.loads(Path(template_path).read_text(encoding="utf-8"))
        return cls(base_url, data, **kw)

    @staticmethod
    def is_available(base_url: str, timeout: float = 2.0) -> bool:
        try:
            r = httpx.get(f"{base_url.rstrip('/')}/system_stats", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def _inject_prompt(self, prompt_text: str) -> dict:
        graph = copy.deepcopy(self.workflow_template)
        replaced = False
        for node in graph.values():
            inputs = node.get("inputs", {})
            if inputs.get("text") == _PROMPT_PLACEHOLDER:
                inputs["text"] = prompt_text
                replaced = True
        if not replaced:
            raise ComfyUIError(
                f"workflow template has no node with text == {_PROMPT_PLACEHOLDER!r}"
            )
        return graph

    def cache_params(self) -> dict[str, str]:
        workflow = json.dumps(
            self.workflow_template,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "provider": self.name,
            "base_url": self.base_url,
            "workflow_sha256": hashlib.sha256(workflow).hexdigest(),
        }

    def generate(self, shot: Shot, assets_dir: Path) -> Path:
        graph = self._inject_prompt(shot.image_prompt)
        assets_dir.mkdir(parents=True, exist_ok=True)
        out = assets_dir / shot.image_file
        request_hash = async_job_request_hash(self.name, self.base_url, {"prompt": graph})
        state = self._resumable_state(out, request_hash)
        if state is not None and state.status == "succeeded" and out.exists():
            return out

        if state is None:
            state = new_async_job_state(
                provider=self.name,
                request_hash=request_hash,
                output_path=out,
                remote_id=str(uuid.uuid4()),
                status="submitting",
            )
            write_async_job_state(out, state)
            state = self._submit(graph, out, state)

        if state.status in {"submitted", "running"}:
            state = transition_async_job(out, state, "running")
        try:
            image_bytes = self._await_image(state.remote_id)
        except _ComfyUIJobFailed:
            transition_async_job(out, state, "failed", error="remote prompt failed")
            raise
        write_bytes_atomic(out, image_bytes)
        transition_async_job(out, state, "succeeded")
        return out

    def _resumable_state(self, out: Path, request_hash: str) -> AsyncJobState | None:
        try:
            state = read_async_job_state(out)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ComfyUIError(
                f"invalid async job state {async_job_state_path(out)}: {type(exc).__name__}"
            ) from exc
        if state is None:
            return None
        if state.provider != self.name or state.request_hash != request_hash:
            if state.status in ACTIVE_JOB_STATUSES:
                raise ComfyUIError(
                    f"active async job conflicts with this request: {async_job_state_path(out)}"
                )
            return None
        if state.status == "failed":
            return None
        if not state.remote_id:
            raise ComfyUIError(
                f"async job has no prompt id; inspect or remove {async_job_state_path(out)}"
            )
        return state

    def _submit(self, graph: dict, out: Path, state: AsyncJobState) -> AsyncJobState:
        try:
            resp = httpx.post(
                f"{self.base_url}/prompt",
                json={"prompt": graph, "prompt_id": state.remote_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            returned_id = str(resp.json()["prompt_id"])
            if returned_id != state.remote_id:
                raise ValueError("ComfyUI returned a different prompt id")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            transition_async_job(out, state, "failed", error=type(exc).__name__)
            raise ComfyUIError("could not connect to ComfyUI; prompt was not submitted") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            terminal = status < 500
            transition_async_job(
                out,
                state,
                "failed" if terminal else "unknown",
                error=f"HTTP {status}",
            )
            if terminal:
                raise ComfyUIError(f"ComfyUI rejected prompt with HTTP {status}") from exc
            return state.model_copy(update={"status": "unknown", "error": f"HTTP {status}"})
        except (httpx.RequestError, KeyError, TypeError, ValueError) as exc:
            return transition_async_job(out, state, "unknown", error=type(exc).__name__)
        return transition_async_job(out, state, "submitted")

    def _await_image(self, prompt_id: str) -> bytes:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            hist = httpx.get(f"{self.base_url}/history/{prompt_id}", timeout=10.0)
            hist.raise_for_status()
            data = hist.json().get(prompt_id)
            if data:
                for node_out in data.get("outputs", {}).values():
                    for img in node_out.get("images", []):
                        return self._fetch_image(img)
                status = data.get("status", {})
                if status.get("status_str") == "error":
                    raise _ComfyUIJobFailed(f"ComfyUI prompt {prompt_id} failed")
            time.sleep(1.0)
        raise ComfyUIError(f"timed out waiting for ComfyUI prompt {prompt_id}")

    def _fetch_image(self, img: dict) -> bytes:
        r = httpx.get(
            f"{self.base_url}/view",
            params={
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            },
            timeout=30.0,
        )
        r.raise_for_status()
        return r.content
