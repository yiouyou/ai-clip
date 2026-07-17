import httpx
import pytest

from ai_clip.core import billing
from ai_clip.core.config import SourceResearchConfig
from ai_clip.core.retry import (
    ExternalCallError,
    FailureCategory,
    RetryPolicy,
    run_with_retry,
)
from ai_clip.source_research import client as research_client


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/v1/call")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("failed", request=request, response=response)


def test_retry_succeeds_after_transient_statuses():
    attempts = []
    delays = []

    def operation():
        attempts.append(1)
        if len(attempts) < 3:
            raise _status_error(503)
        return "ok"

    outcome = run_with_retry(
        operation,
        service="test",
        operation_name="request",
        policy=RetryPolicy(
            max_attempts=3,
            retry_categories=frozenset({FailureCategory.TRANSIENT}),
            base_delay_sec=0.5,
        ),
        sleep=delays.append,
    )

    assert outcome.value == "ok"
    assert outcome.attempts == 3
    assert delays == [0.5, 1.0]


def test_retry_stops_immediately_for_terminal_status():
    calls = []

    with pytest.raises(ExternalCallError) as caught:
        run_with_retry(
            lambda: calls.append(1) or (_ for _ in ()).throw(_status_error(400)),
            service="test",
            operation_name="request",
            policy=RetryPolicy(
                max_attempts=3,
                retry_categories=frozenset({FailureCategory.TRANSIENT}),
            ),
            sleep=lambda _: None,
        )

    assert len(calls) == 1
    assert caught.value.category == FailureCategory.TERMINAL
    assert caught.value.attempts == 1
    assert caught.value.status_code == 400


def test_retry_reports_exhausted_transport_attempts():
    calls = []

    def operation():
        calls.append(1)
        raise httpx.ConnectError("secret response must not leak")

    with pytest.raises(ExternalCallError) as caught:
        run_with_retry(
            operation,
            service="test",
            operation_name="request",
            policy=RetryPolicy(
                max_attempts=2,
                retry_categories=frozenset({FailureCategory.TRANSIENT}),
                base_delay_sec=0,
            ),
            sleep=lambda _: None,
        )

    assert len(calls) == 2
    assert caught.value.category == FailureCategory.TRANSIENT
    assert caught.value.attempts == 2
    assert "secret response" not in str(caught.value)


def test_retry_can_treat_ambiguous_read_timeout_as_terminal():
    calls = []

    def operation():
        calls.append(1)
        raise httpx.ReadTimeout("response interrupted")

    with pytest.raises(ExternalCallError) as caught:
        run_with_retry(
            operation,
            service="paid",
            operation_name="generate",
            policy=RetryPolicy(
                max_attempts=2,
                retry_categories=frozenset({FailureCategory.TIMEOUT}),
            ),
            retry_ambiguous_transport=False,
            sleep=lambda _: None,
        )

    assert len(calls) == 1
    assert caught.value.category == FailureCategory.TERMINAL


def test_tavily_retries_transient_status_and_records_attempts(monkeypatch, tmp_path):
    calls = []

    def fake_post(url, **kwargs):
        request = httpx.Request("POST", url)
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            request=request,
            json={
                "results": [
                    {"title": "source", "url": "https://source.test", "content": "fact"}
                ]
            },
        )

    monkeypatch.setattr(research_client.httpx, "post", fake_post)
    monkeypatch.setattr("ai_clip.core.retry.time.sleep", lambda _: None)
    with billing.account(tmp_path, "research"):
        results = research_client.tavily_search(
            "query",
            SourceResearchConfig(tavily_api_key="key", max_attempts=2),
        )

    assert len(results) == 1
    assert len(calls) == 2
    usage = billing.summarize(tmp_path)
    assert usage["total"]["searches"] == 1
    assert usage["total"]["attempts"] == 2


def test_tavily_does_not_retry_invalid_result(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(1)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"results": [{"score": "not-a-number"}]},
        )

    monkeypatch.setattr(research_client.httpx, "post", fake_post)
    with pytest.raises(research_client.SourceResearchError) as caught:
        research_client.tavily_search(
            "query",
            SourceResearchConfig(tavily_api_key="key", max_attempts=2),
        )

    assert len(calls) == 1
    assert caught.value.category == FailureCategory.INVALID_RESPONSE
