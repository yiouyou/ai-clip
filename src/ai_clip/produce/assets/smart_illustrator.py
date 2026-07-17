"""Smart Illustrator subprocess provider.

This provider keeps Smart Illustrator optional: ai-clip does not import Bun,
Playwright, or Gemini SDKs. It only calls a configured `generate-image.ts`
script when requested and writes the normal `assets/shot_NN.png` output.
"""

from __future__ import annotations

import os
import hashlib
import shutil
import subprocess
from pathlib import Path

from ai_clip.core.config import AssetsConfig
from ai_clip.core.models import Shot


class SmartIllustratorError(RuntimeError):
    pass


class SmartIllustratorProvider:
    name = "smart_illustrator"

    def __init__(self, cfg: AssetsConfig, timeout: float = 600.0) -> None:
        self.cfg = cfg
        self.timeout = timeout
        self.script = self._resolve_script(cfg)

    @staticmethod
    def _resolve_script(cfg: AssetsConfig) -> Path:
        if cfg.smart_illustrator_script:
            return Path(cfg.smart_illustrator_script).expanduser()

        roots = []
        if cfg.smart_illustrator_dir:
            roots.append(Path(cfg.smart_illustrator_dir).expanduser())
        roots.extend([
            Path.home() / ".claude" / "skills" / "smart-illustrator",
            Path.home() / ".codex" / "skills" / "smart-illustrator",
        ])
        for root in roots:
            candidate = root / "scripts" / "generate-image.ts"
            if candidate.exists():
                return candidate
        return roots[0] / "scripts" / "generate-image.ts" if roots else Path()

    @staticmethod
    def is_available(cfg: AssetsConfig) -> bool:
        script = SmartIllustratorProvider._resolve_script(cfg)
        return bool(script and script.exists() and shutil.which("npx"))

    def generate(self, shot: Shot, assets_dir: Path) -> Path:
        if not self.script.exists():
            raise SmartIllustratorError(
                "smart_illustrator provider requested but generate-image.ts was not found; "
                "set AICLIP_SMART_ILLUSTRATOR_DIR or AICLIP_SMART_ILLUSTRATOR_SCRIPT"
            )
        if not shutil.which("npx"):
            raise SmartIllustratorError("smart_illustrator provider requires `npx` on PATH")
        if not shot.image_file:
            raise SmartIllustratorError(f"shot {shot.index} has no image_file")
        if not shot.image_prompt.strip():
            raise SmartIllustratorError(f"shot {shot.index} has no image_prompt")

        assets_dir.mkdir(parents=True, exist_ok=True)
        out = assets_dir / shot.image_file
        prompt_file = assets_dir / "source" / f"{out.stem}_smart_illustrator_prompt.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(shot.image_prompt, encoding="utf-8")

        cmd = [
            "npx", "-y", "bun", str(self.script),
            "--prompt-file", str(prompt_file),
            "--output", str(out),
        ]
        if self.cfg.smart_illustrator_provider:
            cmd.extend(["--provider", self.cfg.smart_illustrator_provider])
        if self.cfg.smart_illustrator_model:
            cmd.extend(["--model", self.cfg.smart_illustrator_model])
        if self.cfg.smart_illustrator_candidates > 1:
            cmd.extend(["--candidates", str(self.cfg.smart_illustrator_candidates)])

        env = os.environ.copy()
        proc = subprocess.run(
            cmd,
            cwd=str(assets_dir),
            env=env,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            raise SmartIllustratorError(
                f"smart_illustrator failed for shot {shot.index}: {detail}"
            )
        if not out.exists():
            raise SmartIllustratorError(
                f"smart_illustrator completed but did not create {out}"
            )
        return out

    def cache_params(self) -> dict[str, str]:
        script_hash = ""
        if self.script.exists():
            script_hash = hashlib.sha256(self.script.read_bytes()).hexdigest()
        return {
            "provider": self.name,
            "script": str(self.script.resolve()) if self.script else "",
            "script_sha256": script_hash,
            "image_provider": self.cfg.smart_illustrator_provider,
            "model": self.cfg.smart_illustrator_model,
            "candidates": str(self.cfg.smart_illustrator_candidates),
        }
