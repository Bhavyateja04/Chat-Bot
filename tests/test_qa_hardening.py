"""Regression tests for defects found during the final QA review.

Each class documents a real bug that was reproduced before being fixed.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.ai.client import LLMClient, LLMError
from app.bot import handlers
from app.bot.session import Session, SessionStore
from app.config import Settings
from app.models.models import SessionState


# --------------------------------------------------------------------------- #
# 1. Race condition: two analyses on one session
# --------------------------------------------------------------------------- #
class TestConcurrentAnalysisGuard:
    @pytest.fixture
    def ready_session(self, parse_fixture):
        session = Session(key=(1, 1))
        session.set_jd(parse_fixture("software_jd.txt"))
        session.add_resume(parse_fixture("strong_software_resume.txt"))
        return session

    async def test_two_concurrent_analyses_do_not_both_run(self, ready_session):
        """A double /analyze must not duplicate the work."""
        first, second = await asyncio.gather(
            handlers.handle_analyze(ready_session),
            handlers.handle_analyze(ready_session),
        )
        reports = [m for m in (first, second) if "ATS SCORE" in "\n".join(m)]
        refusals = [m for m in (first, second) if "already running" in "\n".join(m)]

        assert len(reports) == 1, "Both analyses ran concurrently"
        assert len(refusals) == 1

    async def test_the_flag_is_cleared_afterwards(self, ready_session):
        await handlers.handle_analyze(ready_session)
        assert not ready_session.is_analyzing
        assert ready_session.state == SessionState.RESULTS

    async def test_the_flag_is_cleared_even_when_analysis_fails(
        self, ready_session, monkeypatch
    ):
        async def boom(*args, **kwargs):
            raise RuntimeError("simulated")

        monkeypatch.setattr(handlers, "run_analysis", boom)
        await handlers.handle_analyze(ready_session)

        assert not ready_session.is_analyzing, "Flag stuck after a failure"

    async def test_a_retry_is_possible_after_a_failure(self, ready_session, monkeypatch):
        async def boom(*args, **kwargs):
            raise RuntimeError("simulated")

        monkeypatch.setattr(handlers, "run_analysis", boom)
        await handlers.handle_analyze(ready_session)
        messages = await handlers.handle_analyze(ready_session)
        assert "already running" not in "\n".join(messages)

    async def test_uploads_during_analysis_do_not_change_the_run(
        self, ready_session, parse_fixture, monkeypatch
    ):
        """The analysed set is snapshotted when the run starts."""
        extra = parse_fixture("weak_software_resume.txt")
        original = handlers.run_analysis

        async def slow(jd, resumes, **kwargs):
            # Simulate an upload landing mid-run.
            ready_session.add_resume(extra)
            return await original(jd, resumes, **kwargs)

        monkeypatch.setattr(handlers, "run_analysis", slow)
        messages = await handlers.handle_analyze(ready_session)

        # Only the single snapshotted resume was analysed.
        assert "Arjun Mehta" not in "\n".join(messages)
        assert ready_session.resume_count == 2


# --------------------------------------------------------------------------- #
# 2. Destructive bare-word commands
# --------------------------------------------------------------------------- #
class TestBareWordCommandSafety:
    """'reset my expectations' must not wipe an in-progress session."""

    @pytest.fixture
    def bot(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-token")
        from app.bot.discord_bot import build_bot

        instance = build_bot()

        async def no_op(message):
            return None

        monkeypatch.setattr(instance, "process_commands", no_op)
        return instance

    async def _seed_jd(self, bot, fixture_text):
        from tests.test_discord_bot import FakeAttachment, FakeChannel, FakeMessage

        channel = FakeChannel()
        await bot.on_message(
            FakeMessage(
                channel,
                attachments=[
                    FakeAttachment("software_jd.txt", fixture_text("software_jd.txt"))
                ],
            )
        )
        session = bot.sessions.get(channel.id, 42)
        assert session.has_jd
        return channel, session

    @pytest.mark.parametrize(
        "text",
        [
            "reset my expectations, this pool is weak",
            "status update: sending resumes tomorrow",
            "help me understand the second candidate",
            "start looking at the backend roles",
        ],
    )
    async def test_prose_starting_with_a_command_word_is_not_a_command(
        self, bot, fixture_text, text
    ):
        from tests.test_discord_bot import FakeMessage

        channel, session = await self._seed_jd(bot, fixture_text)
        await bot.on_message(FakeMessage(channel, text))

        assert session.has_jd, f"'{text}' destroyed the session"

    @pytest.mark.parametrize("text", ["reset", "RESET", "/reset", "!reset", "reset."])
    async def test_a_real_reset_still_works(self, bot, fixture_text, text):
        from tests.test_discord_bot import FakeMessage

        channel, session = await self._seed_jd(bot, fixture_text)
        await bot.on_message(FakeMessage(channel, text))

        assert not session.has_jd, f"'{text}' should have reset the session"


# --------------------------------------------------------------------------- #
# 3. Session store memory leak
# --------------------------------------------------------------------------- #
class TestSessionStoreDoesNotLeak:
    def test_abandoned_sessions_are_swept(self):
        store = SessionStore()
        for user_id in range(SessionStore.PURGE_EVERY * 3):
            store.get(1, user_id).last_activity -= 10 ** 6

        store.get(1, 999_999)  # triggers a sweep
        assert len(store) < 10, f"Stale sessions retained: {len(store)}"

    def test_a_session_mid_analysis_is_never_purged(self):
        store = SessionStore()
        session = store.get(1, 1)
        session.is_analyzing = True
        session.last_activity -= 10 ** 6

        store.purge_expired()
        assert store.get(1, 1) is session, "Purged a session that was mid-analysis"


# --------------------------------------------------------------------------- #
# 4. LLM circuit breaker
# --------------------------------------------------------------------------- #
def _settings(**over) -> Settings:
    base = dict(
        llm_api_key="test-key",
        llm_model="m",
        llm_provider="anthropic",
        llm_base_url="https://example.invalid/v1",
        demo_mode=False,
        llm_max_retries=2,
    )
    base.update(over)
    return Settings(**base)


def _patch(monkeypatch, handler):
    original = httpx.AsyncClient.__init__

    def patched(self, *a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        original(self, *a, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


class TestCircuitBreaker:
    async def test_a_billing_error_disables_the_client(self, monkeypatch):
        """No credits is permanent: stop calling out for the rest of the session."""
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(
                400,
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Your credit balance is too low",
                    }
                },
            )

        _patch(monkeypatch, handler)
        client = LLMClient(_settings())

        with pytest.raises(LLMError) as exc:
            await client.complete_json("s", "u")

        assert "credit balance is too low" in str(exc.value)
        assert calls["n"] == 1, "A permanent 400 must not be retried"
        assert not client.available
        assert client.degraded

        with pytest.raises(LLMError):
            await client.complete_json("s", "u")
        assert calls["n"] == 1, "Client kept calling after being disabled"

    async def test_auth_failure_also_trips_the_breaker(self, monkeypatch):
        _patch(monkeypatch, lambda r: httpx.Response(401))
        client = LLMClient(_settings())

        with pytest.raises(LLMError):
            await client.complete_json("s", "u")
        assert client.degraded
        assert "LLM_API_KEY" in client.disabled_reason

    async def test_transient_errors_do_not_trip_the_breaker(self, monkeypatch):
        _patch(monkeypatch, lambda r: httpx.Response(503))
        client = LLMClient(_settings(llm_max_retries=0))

        with pytest.raises(LLMError):
            await client.complete_json("s", "u")
        assert not client.degraded, "A 503 is transient and must not disable the client"

    def test_demo_mode_is_not_reported_as_degraded(self):
        client = LLMClient(_settings(demo_mode=True))
        assert not client.available
        assert not client.degraded, "Demo mode is intentional, not a degradation"

    async def test_error_messages_never_leak_the_api_key(self, monkeypatch):
        secret = "sk-ant-super-secret-value-12345"
        _patch(monkeypatch, lambda r: httpx.Response(401))
        client = LLMClient(_settings(llm_api_key=secret))

        try:
            await client.complete_json("s", "u")
        except LLMError as exc:
            assert secret not in str(exc)
        assert secret not in (client.disabled_reason or "")


class TestDegradationIsSurfaced:
    async def test_the_user_is_told_when_the_llm_was_unavailable(
        self, parse_fixture, monkeypatch
    ):
        """A silently degraded analysis is worse than a noisy one."""
        _patch(
            monkeypatch,
            lambda r: httpx.Response(
                400, json={"error": {"message": "Your credit balance is too low"}}
            ),
        )
        client = LLMClient(_settings())

        session = Session(key=(1, 1))
        session.set_jd(parse_fixture("software_jd.txt"))
        session.add_resume(parse_fixture("strong_software_resume.txt"))

        messages = await handlers.handle_analyze(session, client=client)
        blob = "\n".join(messages)

        assert "ATS SCORE" in blob, "The analysis must still complete"
        assert "AI provider was unavailable" in blob
        assert "credit balance is too low" in blob

    async def test_no_such_note_when_running_in_demo_mode(self, parse_fixture):
        session = Session(key=(1, 2))
        session.set_jd(parse_fixture("software_jd.txt"))
        session.add_resume(parse_fixture("strong_software_resume.txt"))

        messages = await handlers.handle_analyze(session, client=LLMClient())
        assert "AI provider was unavailable" not in "\n".join(messages)
