from __future__ import annotations

from contextlib import AbstractContextManager
import ctypes
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

from ai_clip.core.artifacts import write_text_atomic


class RunLock(AbstractContextManager["RunLock"]):
    def __init__(
        self,
        path: Path,
        label: str,
        *,
        stale_after_minutes: int = 180,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.path = path
        self.label = label
        self.stale_after_minutes = stale_after_minutes
        self.metadata = metadata or {}
        self.acquired = False

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not _lock_is_stale(self.path, self.stale_after_minutes):
            raise RuntimeError(f"{self.label}: {self.path}")
        payload = {
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            **self.metadata,
        }
        write_text_atomic(self.path, json.dumps(payload, indent=2), encoding="utf-8")
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self.acquired:
            self.path.unlink(missing_ok=True)
        return False


def _lock_is_stale(path: Path, stale_after_minutes: int) -> bool:
    data = _read_json(path)
    if not isinstance(data, dict):
        return True
    started_at = str(data.get("started_at") or "")
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return True
    age = datetime.now(timezone.utc) - started.astimezone(timezone.utc)
    if age > timedelta(minutes=stale_after_minutes):
        return True
    pid = data.get("pid")
    return isinstance(pid, int) and not _pid_exists(pid)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_exists(pid)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _windows_pid_exists(pid: int) -> bool:
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information,
        False,
        pid,
    )
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True
