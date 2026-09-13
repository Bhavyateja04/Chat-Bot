"""Discord plumbing.

All conversation logic lives in ``handlers.py``; this module only translates
between Discord events and those handlers.

Both interaction styles are supported:

* **Slash commands** (``/start``, ``/help``, ``/status``, ``/analyze``, ``/reset``,
  ``/jd``) - registered with Discord and synced on startup.
* **Plain messages** - file uploads, disambiguation replies, and ``!``-prefixed
  fallbacks for the same commands, so the bot still works if slash commands have
  not propagated yet.
"""

from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands
from discord.ext import commands

from app.ai.client import LLMClient
from app.bot import handlers
from app.bot.session import SessionStore
from app.config import get_settings
from app.models.models import SessionState
from app.parsers import loader

logger = logging.getLogger(__name__)

# Discord attachment downloads are capped so one huge file cannot stall the bot.
_DOWNLOAD_TIMEOUT = 30.0


class ResumeMatchBot(commands.Bot):
    """The ResumeMatch AI Discord bot."""

    def __init__(self) -> None:
        settings = get_settings()
        intents = discord.Intents.default()
        # Required to read uploaded files and plain-text replies. This is a
        # privileged intent - enable "Message Content Intent" in the Discord
        # developer portal (see README).
        intents.message_content = True

        super().__init__(
            command_prefix=settings.discord_command_prefix,
            intents=intents,
            help_command=None,
            description="Compare resumes against a job description.",
        )
        self.settings = settings
        self.sessions = SessionStore()
        self.llm = LLMClient(settings)

    # --- lifecycle ------------------------------------------------------ #
    async def setup_hook(self) -> None:
        register_commands(self)
        try:
            synced = await self.tree.sync()
            logger.info("Synced %d slash command(s).", len(synced))
        except Exception:
            logger.exception(
                "Could not sync slash commands. The '!' prefix commands still work."
            )

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (id=%s)", self.user, getattr(self.user, "id", "?"))
        logger.info("Connected to %d guild(s).", len(self.guilds))
        mode = "DEMO (no LLM calls)" if not self.llm.available else (
            f"LLM: {self.settings.llm_provider}/{self.settings.llm_model}"
        )
        logger.info("Analysis mode - %s", mode)
        try:
            await self.change_presence(
                activity=discord.Game(name="/start to match resumes")
            )
        except Exception:
            logger.debug("Could not set presence.", exc_info=True)

    async def on_command_error(self, context, exception) -> None:
        if isinstance(exception, commands.CommandNotFound):
            return
        logger.exception("Command error", exc_info=exception)
        await safe_send(context.channel, "❌ Something went wrong running that command.")

    # --- message handling ----------------------------------------------- #
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        channel_id = message.channel.id
        session = self.sessions.get(channel_id, message.author.id)
        content = (message.content or "").strip()

        # 1. File uploads take priority.
        if message.attachments:
            await self._handle_attachments(message, session)
            return

        # 2. Resolve a pending "is this a JD or a resume?" question.
        if session.state == SessionState.AWAITING_DISAMBIGUATION and content:
            replies = handlers.handle_disambiguation(session, content)
            if replies:
                await send_all(message.channel, replies)
                return

        # 3. Plain-text command fallbacks (works even before slash commands sync).
        #    A bare word only counts as a command when it is the ENTIRE message:
        #    "reset my expectations about this candidate" must not wipe the
        #    session. An explicit /reset or !reset always works.
        prefixed = content.startswith(("/", "!"))
        lowered = content.lower().lstrip("!/").strip()
        # Strip trailing punctuation so "reset." still matches "reset".
        simple = lowered.split()[0].strip(".!?,;:") if lowered else ""
        bare_command = lowered.strip(" .!?") in {"start", "help", "reset", "status"}
        if prefixed or bare_command:
            if simple in {"start", "begin"}:
                await send_all(message.channel, handlers.handle_start(session))
                return
            if simple == "help":
                await send_all(message.channel, handlers.handle_help())
                return
            if simple == "reset":
                await send_all(message.channel, handlers.handle_reset(session))
                return
            if simple == "status":
                await send_all(message.channel, handlers.handle_status(session))
                return
            if simple == "analyze" or simple == "analyse":
                await self._run_analysis(message.channel, session)
                return
            if simple == "jd":
                pasted = content.split(None, 1)[1] if len(content.split(None, 1)) > 1 else ""
                await send_all(
                    message.channel, handlers.handle_pasted_jd(session, pasted)
                )
                return

        # 4. A long paste with no JD yet is treated as a pasted job description.
        if not session.has_jd and len(content) > 400:
            await send_all(message.channel, handlers.handle_pasted_jd(session, content))
            return

        # 5. Greetings, thanks and "what can you do" always get an answer.
        if content:
            reply = handlers.handle_chat(session, content)
            if reply:
                await send_all(message.channel, reply)
                return

            # Anything else only gets a nudge when this user has work in
            # progress, is in a DM, or mentioned the bot - otherwise the bot
            # would reply to every unrelated line in a busy channel.
            if self._should_answer(message, session):
                await send_all(
                    message.channel, handlers.handle_unrecognised(session, content)
                )
                return

        await self.process_commands(message)

    def _should_answer(self, message: discord.Message, session) -> bool:
        """Whether unrecognised text deserves a reply in this context."""
        if isinstance(message.channel, discord.DMChannel):
            return True
        if self.user is not None and self.user in getattr(message, "mentions", []):
            return True
        return session.state != SessionState.IDLE or session.file_count > 0

    async def _handle_attachments(
        self, message: discord.Message, session
    ) -> None:
        """Download, validate and register every attachment on a message."""
        uploads: list[tuple[str, bytes]] = []
        errors: list[str] = []

        async with typing(message.channel):
            for attachment in message.attachments:
                if not loader.is_supported(attachment.filename):
                    errors.append(loader.unsupported_message(attachment.filename))
                    continue
                if attachment.size > self.settings.max_file_size_bytes:
                    errors.append(
                        f"⚠️ **`{attachment.filename}`** is "
                        f"{attachment.size / 1024 / 1024:.1f} MB, over the "
                        f"{self.settings.max_file_size_mb:.0f} MB limit."
                    )
                    continue
                try:
                    data = await asyncio.wait_for(
                        attachment.read(), timeout=_DOWNLOAD_TIMEOUT
                    )
                    uploads.append((attachment.filename, data))
                except asyncio.TimeoutError:
                    errors.append(
                        f"⚠️ Timed out downloading **`{attachment.filename}`**. "
                        "Please try uploading it again."
                    )
                except discord.HTTPException as exc:
                    logger.warning("Attachment download failed: %s", exc)
                    errors.append(
                        f"⚠️ Discord wouldn't let me download "
                        f"**`{attachment.filename}`**. Please try again."
                    )

        if errors:
            await send_all(message.channel, errors)
        if not uploads:
            return

        messages, ready = handlers.handle_uploads(session, uploads)
        await send_all(message.channel, messages)

        # Auto-run once a JD and at least one resume are present, so the demo
        # flows without the user having to remember /analyze.
        if ready:
            await self._run_analysis(message.channel, session)

    async def _run_analysis(self, channel, session) -> None:
        """Run the analysis with live progress updates."""
        if not session.is_ready:
            await send_all(
                channel,
                handlers.handle_status(session)
                if session.has_jd or session.resume_count
                else [
                    "⚠️ Upload a job description and at least one resume first. "
                    "Run `/help` to see how."
                ],
            )
            return

        count = session.resume_count
        await safe_send(
            channel,
            f"🔍 **Analyzing {count} resume{'s' if count != 1 else ''} against the "
            f"job description…**",
        )

        progress_message = None

        async def progress(text: str) -> None:
            nonlocal progress_message
            try:
                if progress_message is None:
                    progress_message = await channel.send(text)
                else:
                    await progress_message.edit(content=text)
            except discord.HTTPException:
                logger.debug("Progress update failed", exc_info=True)

        async with typing(channel):
            messages = await handlers.handle_analyze(
                session, progress=progress, client=self.llm
            )

        if progress_message is not None:
            try:
                await progress_message.delete()
            except discord.HTTPException:
                pass

        await send_all(channel, messages)


# --------------------------------------------------------------------------- #
# Slash commands
# --------------------------------------------------------------------------- #
def register_commands(bot: ResumeMatchBot) -> None:
    """Register the application (slash) commands on the bot's command tree."""

    @bot.tree.command(name="start", description="Start a new resume matching session")
    async def start(interaction: discord.Interaction) -> None:
        session = bot.sessions.get(interaction.channel_id, interaction.user.id)
        await interaction.response.send_message(handlers.handle_start(session)[0])

    @bot.tree.command(name="help", description="How to use ResumeMatch AI")
    async def help_command(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(handlers.handle_help()[0])

    @bot.tree.command(name="reset", description="Clear the current session")
    async def reset(interaction: discord.Interaction) -> None:
        session = bot.sessions.get(interaction.channel_id, interaction.user.id)
        await interaction.response.send_message(handlers.handle_reset(session)[0])

    @bot.tree.command(name="status", description="Show what has been uploaded so far")
    async def status(interaction: discord.Interaction) -> None:
        session = bot.sessions.get(interaction.channel_id, interaction.user.id)
        await interaction.response.send_message(handlers.handle_status(session)[0])

    @bot.tree.command(name="analyze", description="Analyse the uploaded resumes")
    async def analyze(interaction: discord.Interaction) -> None:
        session = bot.sessions.get(interaction.channel_id, interaction.user.id)
        if not session.is_ready:
            await interaction.response.send_message(
                handlers.handle_status(session)[0], ephemeral=True
            )
            return
        # Analysis takes longer than Discord's 3 second interaction window.
        await interaction.response.send_message("🔍 **Starting analysis…**")
        await bot._run_analysis(interaction.channel, session)

    @bot.tree.command(name="jd", description="Paste a job description as text")
    @app_commands.describe(text="The full job description text")
    async def jd(interaction: discord.Interaction, text: str) -> None:
        session = bot.sessions.get(interaction.channel_id, interaction.user.id)
        await interaction.response.send_message(
            handlers.handle_pasted_jd(session, text)[0]
        )


# --------------------------------------------------------------------------- #
# Send helpers
# --------------------------------------------------------------------------- #
def typing(channel):
    """``channel.typing()`` when available, otherwise a no-op context manager."""
    typing_method = getattr(channel, "typing", None)
    if typing_method is None:
        return _NullContext()
    try:
        return typing_method()
    except Exception:
        return _NullContext()


class _NullContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return False


async def safe_send(channel, content: str) -> None:
    """Send one message, tolerating Discord failures and over-long content."""
    if not content:
        return
    try:
        if len(content) <= 2000:
            await channel.send(content)
            return
        from app.formatting.discord_format import pack_messages

        for chunk in pack_messages([content]):
            await channel.send(chunk)
            await asyncio.sleep(0.25)
    except discord.HTTPException as exc:
        logger.warning("Failed to send message: %s", exc)
        try:
            await channel.send(
                "⚠️ I couldn't send part of the response (Discord rejected it)."
            )
        except discord.HTTPException:
            logger.error("Could not send the fallback error message either.")


async def send_all(channel, messages: list[str]) -> None:
    """Send a sequence of messages in order, with a small gap to avoid rate limits."""
    for index, message in enumerate(messages):
        await safe_send(channel, message)
        if index < len(messages) - 1:
            await asyncio.sleep(0.3)


def build_bot() -> ResumeMatchBot:
    return ResumeMatchBot()
