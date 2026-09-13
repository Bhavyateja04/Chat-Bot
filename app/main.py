"""Application entry point: logging setup, config checks, and bot startup."""

from __future__ import annotations

import logging
import sys

from app.config import get_settings


def configure_logging(level: str = "INFO") -> None:
    """Console logging that survives Windows code pages.

    The default Windows console is cp1252 and raises on emoji, so stdout is
    reconfigured to UTF-8 where the runtime supports it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # discord.py is chatty at INFO.
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("discord.http").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Voice support is irrelevant here; suppress the PyNaCl notice.
    logging.getLogger("discord.client").setLevel(logging.ERROR)


def preflight(settings) -> list[str]:
    """Human-readable startup problems. Empty list means good to go."""
    problems: list[str] = []
    if not settings.discord_bot_token:
        problems.append(
            "DISCORD_BOT_TOKEN is not set. Copy .env.example to .env and paste your "
            "bot token from https://discord.com/developers/applications"
        )
    if not settings.llm_api_key and not settings.demo_mode:
        problems.append(
            "LLM_API_KEY is not set and DEMO_MODE is false. Either add an API key, "
            "or set DEMO_MODE=true in .env to run the deterministic analyser with "
            "no LLM calls."
        )
    return problems


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger("resumematch")

    logger.info("ResumeMatch AI starting up…")
    logger.info("Configuration: %s", settings.safe_summary())

    problems = preflight(settings)
    if problems:
        for problem in problems:
            logger.error("Startup problem: %s", problem)
        if not settings.discord_bot_token:
            logger.error("Cannot start without a Discord bot token. Exiting.")
            return 1

    if settings.demo_mode or not settings.llm_api_key:
        logger.warning(
            "Running in DEMO MODE - analysis uses the deterministic engine only, "
            "with no LLM calls."
        )

    # Imported late so that a missing discord.py gives a clean message.
    try:
        from app.bot.discord_bot import build_bot
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        logger.error("Run: pip install -r requirements.txt")
        return 1

    bot = build_bot()

    try:
        bot.run(settings.discord_bot_token, log_handler=None)
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        return 0
    except Exception as exc:
        message = str(exc).lower()
        if "improper token" in message or "unauthorized" in message or "401" in message:
            logger.error(
                "Discord rejected the bot token. Check DISCORD_BOT_TOKEN in .env - "
                "it must be the BOT token, not the client secret or application ID."
            )
        elif "privileged" in message or "intents" in message:
            logger.error(
                "Discord rejected the connection because a privileged intent is not "
                "enabled. Open your application in the Discord developer portal, go "
                "to Bot, and switch ON 'Message Content Intent'."
            )
        else:
            logger.exception("The bot stopped with an unexpected error.")
        return 1
    return 0
