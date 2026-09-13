"""Discord event handling, driven through stub objects.

``on_message`` only touches a small, well-defined surface of the Discord API, so
the whole conversation can be exercised without a network connection. This is
what verifies the integration wiring short of a live token.
"""

from __future__ import annotations

import pytest

from app.formatting.discord_format import pack_messages


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class FakeChannel:
    def __init__(self, channel_id: int = 1) -> None:
        self.id = channel_id
        self.sent: list[str] = []

    async def send(self, content=None, **kwargs):
        text = content if content is not None else kwargs.get("content", "")
        assert len(text) <= 2000, f"Message exceeds Discord's limit: {len(text)}"
        self.sent.append(text)
        return FakeSentMessage(text)

    def typing(self):
        return FakeTyping()

    @property
    def text(self) -> str:
        return "\n".join(self.sent)


class FakeSentMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.deleted = False

    async def edit(self, content=None, **kwargs):
        self.content = content or ""
        return self

    async def delete(self):
        self.deleted = True


class FakeTyping:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return False


class FakeAuthor:
    def __init__(self, user_id: int = 42, bot: bool = False) -> None:
        self.id = user_id
        self.bot = bot


class FakeAttachment:
    def __init__(self, filename: str, data: bytes) -> None:
        self.filename = filename
        self._data = data
        self.size = len(data)

    async def read(self) -> bytes:
        return self._data


class FakeMessage:
    def __init__(self, channel, content="", attachments=None, author=None) -> None:
        self.channel = channel
        self.content = content
        self.attachments = attachments or []
        self.author = author or FakeAuthor()


@pytest.fixture
def bot(monkeypatch):
    """A bot instance that never connects to Discord."""
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-token")
    from app.bot.discord_bot import build_bot

    instance = build_bot()

    async def no_op_process(message):
        return None

    # process_commands would try to reach the gateway.
    monkeypatch.setattr(instance, "process_commands", no_op_process)
    return instance


@pytest.fixture
def channel():
    return FakeChannel()


# --------------------------------------------------------------------------- #
# Commands over plain messages
# --------------------------------------------------------------------------- #
class TestTextCommands:
    async def test_start(self, bot, channel):
        await bot.on_message(FakeMessage(channel, "/start"))
        assert "Welcome to ResumeMatch AI" in channel.text

    async def test_help(self, bot, channel):
        await bot.on_message(FakeMessage(channel, "/help"))
        assert "ATS Score" in channel.text

    async def test_status_when_empty(self, bot, channel):
        await bot.on_message(FakeMessage(channel, "/status"))
        assert "Nothing uploaded" in channel.text

    async def test_reset(self, bot, channel):
        await bot.on_message(FakeMessage(channel, "/reset"))
        assert "cleared" in channel.text.lower()

    async def test_bang_prefix_also_works(self, bot, channel):
        await bot.on_message(FakeMessage(channel, "!help"))
        assert "ATS Score" in channel.text

    async def test_bot_messages_are_ignored(self, bot, channel):
        await bot.on_message(
            FakeMessage(channel, "/start", author=FakeAuthor(bot=True))
        )
        assert channel.sent == []


# --------------------------------------------------------------------------- #
# The full upload -> analysis flow
# --------------------------------------------------------------------------- #
class TestUploadFlow:
    async def test_jd_upload_is_recognised(self, bot, channel, fixture_text):
        await bot.on_message(
            FakeMessage(
                channel,
                attachments=[
                    FakeAttachment("software_jd.txt", fixture_text("software_jd.txt"))
                ],
            )
        )
        assert "Job Description received" in channel.text

    async def test_full_flow_with_three_resumes(self, bot, channel, fixture_text):
        # 1. Job description
        await bot.on_message(
            FakeMessage(
                channel,
                attachments=[
                    FakeAttachment("software_jd.txt", fixture_text("software_jd.txt"))
                ],
            )
        )
        # 2. Three resumes in a single message - analysis runs automatically.
        await bot.on_message(
            FakeMessage(
                channel,
                attachments=[
                    FakeAttachment(name, fixture_text(name))
                    for name in (
                        "strong_software_resume.txt",
                        "weak_software_resume.txt",
                        "mechanical_resume.txt",
                    )
                ],
            )
        )

        transcript = channel.text
        assert "CANDIDATE COMPARISON" in transcript
        assert "Best aligned candidate" in transcript
        assert "ATS SCORE" in transcript
        assert "RECOMMENDED COURSES" in transcript
        assert "ROLE MISMATCH DETECTED" in transcript
        assert "Analysis complete" in transcript

    async def test_mechanical_jd_vs_software_resume_end_to_end(
        self, bot, channel, fixture_text
    ):
        """The recruiter's headline scenario, driven through the bot."""
        await bot.on_message(
            FakeMessage(
                channel,
                attachments=[
                    FakeAttachment("mechanical_jd.txt", fixture_text("mechanical_jd.txt"))
                ],
            )
        )
        await bot.on_message(
            FakeMessage(
                channel,
                attachments=[
                    FakeAttachment(
                        "strong_software_resume.txt",
                        fixture_text("strong_software_resume.txt"),
                    )
                ],
            )
        )

        transcript = channel.text
        assert "ROLE MISMATCH DETECTED" in transcript
        assert "Mechanical" in transcript

    async def test_unsupported_file_is_rejected_politely(self, bot, channel):
        await bot.on_message(
            FakeMessage(channel, attachments=[FakeAttachment("photo.png", b"\x89PNG")])
        )
        assert "unsupported" in channel.text.lower()

    async def test_oversized_attachment_is_rejected_before_download(
        self, bot, channel, monkeypatch
    ):
        monkeypatch.setattr(bot.settings, "max_file_size_mb", 0.0001)
        await bot.on_message(
            FakeMessage(channel, attachments=[FakeAttachment("big.txt", b"x" * 10_000)])
        )
        assert "limit" in channel.text.lower()

    async def test_analyze_without_uploads_is_guided(self, bot, channel):
        await bot.on_message(FakeMessage(channel, "/analyze"))
        assert "upload" in channel.text.lower()

    async def test_reset_between_analyses_clears_state(
        self, bot, channel, fixture_text
    ):
        await bot.on_message(
            FakeMessage(
                channel,
                attachments=[
                    FakeAttachment("software_jd.txt", fixture_text("software_jd.txt"))
                ],
            )
        )
        await bot.on_message(FakeMessage(channel, "/reset"))
        await bot.on_message(FakeMessage(channel, "/status"))

        assert "Nothing uploaded" in channel.sent[-1]


# --------------------------------------------------------------------------- #
# Isolation between users
# --------------------------------------------------------------------------- #
class TestUserIsolation:
    async def test_two_users_in_one_channel_do_not_share_files(
        self, bot, channel, fixture_text
    ):
        alice = FakeAuthor(user_id=1)
        bob = FakeAuthor(user_id=2)

        await bot.on_message(
            FakeMessage(
                channel,
                attachments=[
                    FakeAttachment("software_jd.txt", fixture_text("software_jd.txt"))
                ],
                author=alice,
            )
        )
        channel.sent.clear()

        await bot.on_message(FakeMessage(channel, "/status", author=bob))
        assert "Nothing uploaded" in channel.text


# --------------------------------------------------------------------------- #
# Disambiguation over Discord
# --------------------------------------------------------------------------- #
class TestDisambiguationFlow:
    async def test_ambiguous_document_is_queried_then_resolved(
        self, bot, channel, monkeypatch
    ):
        from app.bot import handlers
        from app.models.models import DocumentKind

        # Force the classifier to be unsure about this document.
        monkeypatch.setattr(
            handlers, "classify_document", lambda text, name: (DocumentKind.UNKNOWN, 0.5)
        )
        # The document is genuinely resume-or-JD shaped; only the *kind* is unclear.
        monkeypatch.setattr(handlers, "has_enough_signal", lambda text, name: True)

        content = b"Some document text that is long enough to be parsed properly. " * 5
        await bot.on_message(
            FakeMessage(channel, attachments=[FakeAttachment("mystery.txt", content)])
        )
        assert "can't tell" in channel.text.lower()

        await bot.on_message(FakeMessage(channel, "jd"))
        assert "job description" in channel.sent[-1].lower()


# --------------------------------------------------------------------------- #
# Send helpers
# --------------------------------------------------------------------------- #
class TestSendHelpers:
    async def test_over_long_content_is_split_before_sending(self, channel):
        from app.bot.discord_bot import safe_send

        await safe_send(channel, "x" * 5000)
        assert len(channel.sent) > 1
        assert all(len(m) <= 2000 for m in channel.sent)

    async def test_send_all_preserves_order(self, channel):
        from app.bot.discord_bot import send_all

        await send_all(channel, ["first", "second", "third"])
        assert channel.sent == ["first", "second", "third"]

    def test_packing_never_exceeds_the_limit(self):
        blocks = ["line " * 200 for _ in range(30)]
        for message in pack_messages(blocks):
            assert len(message) <= 2000


class TestSmallTalk:
    """Plain conversational messages must get a reply, not silence."""

    @pytest.mark.parametrize(
        "greeting", ["hi", "Hi", "hello", "hey", "HELLO", "yo", "good morning", "hi there"]
    )
    async def test_greetings_get_a_reply(self, bot, channel, greeting):
        await bot.on_message(FakeMessage(channel, greeting))

        assert channel.sent, f"'{greeting}' got no reply"
        assert "ResumeMatch AI" in channel.text
        assert "job description" in channel.text.lower()

    async def test_greeting_reply_reflects_session_state(
        self, bot, channel, fixture_text
    ):
        await bot.on_message(
            FakeMessage(
                channel,
                attachments=[
                    FakeAttachment("software_jd.txt", fixture_text("software_jd.txt"))
                ],
            )
        )
        channel.sent.clear()
        await bot.on_message(FakeMessage(channel, "hi"))

        assert "upload **one or more resumes**" in channel.text

    async def test_thanks_gets_a_reply(self, bot, channel):
        await bot.on_message(FakeMessage(channel, "thanks"))
        assert "welcome" in channel.text.lower()

    async def test_goodbye_gets_a_reply(self, bot, channel):
        await bot.on_message(FakeMessage(channel, "bye"))
        assert "goodbye" in channel.text.lower()

    async def test_what_can_you_do_returns_help(self, bot, channel):
        await bot.on_message(FakeMessage(channel, "what can you do"))
        assert "ATS Score" in channel.text

    async def test_unrecognised_text_is_ignored_in_an_idle_channel(self, bot, channel):
        """The bot must not reply to every unrelated line in a busy server."""
        await bot.on_message(FakeMessage(channel, "did anyone watch the match"))
        assert channel.sent == []

    async def test_unrecognised_text_is_answered_during_an_active_session(
        self, bot, channel, fixture_text
    ):
        await bot.on_message(
            FakeMessage(
                channel,
                attachments=[
                    FakeAttachment("software_jd.txt", fixture_text("software_jd.txt"))
                ],
            )
        )
        channel.sent.clear()
        await bot.on_message(FakeMessage(channel, "what now"))

        assert "didn't quite catch that" in channel.text
        assert "Next step" in channel.text

    async def test_greeting_does_not_disturb_a_pending_disambiguation(
        self, bot, channel, monkeypatch
    ):
        """A pending jd/resume question still takes priority over small talk."""
        from app.bot import handlers
        from app.models.models import DocumentKind

        monkeypatch.setattr(
            handlers, "classify_document", lambda text, name: (DocumentKind.UNKNOWN, 0.5)
        )
        monkeypatch.setattr(handlers, "has_enough_signal", lambda text, name: True)

        await bot.on_message(
            FakeMessage(
                channel,
                attachments=[FakeAttachment("mystery.txt", b"Some document text. " * 30)],
            )
        )
        channel.sent.clear()
        await bot.on_message(FakeMessage(channel, "hi"))

        # The disambiguation handler answers first and re-asks the question.
        assert "reply" in channel.text.lower()
