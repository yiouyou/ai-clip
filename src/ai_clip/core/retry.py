from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
import json
import time
from typing import Generic, TypeVar

import httpx


class FailureCategory(StrEnum):
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    TRANSIENT = "transient"
    INVALID_RESPONSE = "invalid_response"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class ClassifiedFailure:
    category: FailureCategory
    detail: str
    status_code: int | None = None


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    retry_categories: frozenset[FailureCategory] = field(default_factory=frozenset)
    base_delay_sec: float = 1.0
    max_delay_sec: float = 4.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("retry max_attempts must be at least 1")
        if self.base_delay_sec < 0 or self.max_delay_sec < 0:
            raise ValueError("retry delays cannot be negative")

    def delay_after(self, attempt: int) -> float:
        return min(self.base_delay_sec * (2 ** max(attempt - 1, 0)), self.max_delay_sec)


ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class CallOutcome(Generic[ResultT]):
    value: ResultT
    attempts: int


class ExternalCallError(RuntimeError):
    def __init__(
        self,
        message: str | None = None,
        *,
        service: str = "external",
        operation: str = "call",
        category: FailureCategory = FailureCategory.TERMINAL,
        attempts: int = 0,
        status_code: int | None = None,
        detail: str = "",
    ) -> None:
        self.service = service
        self.operation = operation
        self.category = category
        self.attempts = attempts
        self.status_code = status_code
        self.detail = _safe_detail(detail)
        if message is None:
            status = f" status={status_code}" if status_code is not None else ""
            message = (
                f"{service} {operation} failed: category={category.value} "
                f"attempts={attempts}{status} detail={self.detail or 'unavailable'}"
            )
        super().__init__(message)


ErrorT = TypeVar("ErrorT", bound=ExternalCallError)


def run_with_retry(
    operation: Callable[[], ResultT],
    *,
    service: str,
    operation_name: str,
    policy: RetryPolicy,
    error_type: type[ErrorT] = ExternalCallError,
    retry_ambiguous_transport: bool = True,
    classifier: Callable[[Exception], ClassifiedFailure] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> CallOutcome[ResultT]:
    classify = classifier or (
        lambda exc: classify_http_failure(
            exc,
            retry_ambiguous_transport=retry_ambiguous_transport,
        )
    )
    sleeper = sleep or time.sleep
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return CallOutcome(value=operation(), attempts=attempt)
        except Exception as exc:
            failure = classify(exc)
            retryable = failure.category in policy.retry_categories
            if not retryable or attempt >= policy.max_attempts:
                raise error_type(
                    service=service,
                    operation=operation_name,
                    category=failure.category,
                    attempts=attempt,
                    status_code=failure.status_code,
                    detail=failure.detail,
                ) from exc
            sleeper(policy.delay_after(attempt))
    raise AssertionError("retry loop exited without a result")


def classify_http_failure(
    exc: Exception,
    *,
    retry_ambiguous_transport: bool = True,
) -> ClassifiedFailure:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in {401, 403}:
            category = FailureCategory.AUTHENTICATION
        elif status == 429:
            category = FailureCategory.RATE_LIMIT
        elif status in {408, 504}:
            category = FailureCategory.TIMEOUT
        elif status >= 500:
            category = FailureCategory.TRANSIENT
        else:
            category = FailureCategory.TERMINAL
        return ClassifiedFailure(category, f"HTTP {status}", status)
    if isinstance(exc, (httpx.ConnectTimeout, httpx.PoolTimeout)):
        return ClassifiedFailure(FailureCategory.TIMEOUT, exc.__class__.__name__)
    if isinstance(exc, httpx.ConnectError):
        return ClassifiedFailure(FailureCategory.TRANSIENT, exc.__class__.__name__)
    if isinstance(exc, httpx.TimeoutException):
        category = (
            FailureCategory.TIMEOUT
            if retry_ambiguous_transport
            else FailureCategory.TERMINAL
        )
        return ClassifiedFailure(category, exc.__class__.__name__)
    if isinstance(exc, httpx.TransportError):
        category = (
            FailureCategory.TRANSIENT
            if retry_ambiguous_transport
            else FailureCategory.TERMINAL
        )
        return ClassifiedFailure(category, exc.__class__.__name__)
    if isinstance(exc, json.JSONDecodeError):
        return ClassifiedFailure(FailureCategory.INVALID_RESPONSE, "invalid JSON response")
    if isinstance(exc, (KeyError, TypeError, ValueError)):
        return ClassifiedFailure(FailureCategory.INVALID_RESPONSE, exc.__class__.__name__)
    return ClassifiedFailure(FailureCategory.TERMINAL, exc.__class__.__name__)


def _safe_detail(detail: str) -> str:
    return " ".join(str(detail).split())[:160]
