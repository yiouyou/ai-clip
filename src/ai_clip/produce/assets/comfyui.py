"""ComfyUI API provider: submit a workflow JSON template with the shot's prompt
injected, poll for completion, and save the resulting image.

The workflow template is a ComfyUI "API format" graph exported from the UI. We
only swap the positive-prompt text into nodes tagged with AICLIP_PROMPT.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import httpx

from ai_clip.core.models import Shot

_PROMPT_PLACEHOLDER = "AICLIP_PROMPT"


class ComfyUIError(RuntimeError):
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

    def generate(self, shot: Shot, assets_dir: Path) -> Path:
        graph = self._inject_prompt(shot.image_prompt)
        resp = httpx.post(
            f"{self.base_url}/prompt", json={"prompt": graph}, timeout=self.timeout
        )
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
        image_bytes = self._await_image(prompt_id)

        assets_dir.mkdir(parents=True, exist_ok=True)
        out = assets_dir / shot.image_file
        out.write_bytes(image_bytes)
        return out

    def _await_image(self, prompt_id: str) -> bytes:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            hist = httpx.get(f"{self.base_url}/history/{prompt_id}", timeout=10.0)
            data = hist.json().get(prompt_id)
            if data:
                for node_out in data.get("outputs", {}).values():
                    for img in node_out.get("images", []):
                        return self._fetch_image(img)
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
