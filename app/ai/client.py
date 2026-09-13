"""Provider-agnostic LLM client.

Speaks two wire formats:

* ``anthropic`` - the Claude Messages API (``POST {base_url}/messages``)
* ``openai``    - any OpenAI-compatible chat completions endpoint
  (``POST {base_url}/chat/completions``), which covers OpenAI, Groq, Together,
  OpenRouter, Ollama and most gateways.

Switching provider is a ``.env`` change, no code change:

    LLM_PROVIDER=openai
    LLM_BASE_URL=https://api.openai.com/v1
    LLM_MODEL=gpt-4o-mini

The client returns parsed JSON only. Malformed responses are repaired where it is
safe to do so, retried once, and otherwise raise ``LLMError`` so the caller can
fall back to the deterministic extractor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Any failure to obtain usable JSON from the model."""


class LLMNotConfigured(LLMError):
    """No API key configured, or DEMO_MODE is on."""


# --------------------------------------------------------------------------- #
# JSON extraction / repair
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Handles the three things models actually do wrong: wrapping JSON in code
    fences, adding a sentence before or after it, and leaving a trailing comma.
    """
    if not raw or not raw.strip():
        raise LLMError("The model returned an empty response.")

    candidates: list[str] = []

    fenced = _FENCE_RE.search(raw)
    if fenced:
        candidates.append(fenced.group(1).strip())

    stripped = raw.strip()
    candidates.append(stripped)

    # Widest balanced-looking span between the first { and the last }.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _repair(candidate)
            if repaired is None:
                continue
            parsed = repaired
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return {"analyses": parsed}

    raise LLMError("The model response did not contain valid JSON.")


def _repair(text: str) -> dict[str, Any] | None:
    """Conservative repairs only - never guess at missing content."""
    # Remove trailing commas before a closing brace/bracket.
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    # Strip // and /* */ comments some models emit.
    fixed = re.sub(r"//[^\n\"]*", "", fixed)
    fixed = re.sub(r"/\*.*?\*/", "", fixed, flags=re.DOTALL)
    try:
        result = json.loads(fixed)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass

    # Truncated response (the model hit its token limit mid-object).
    patched = _close_open_structures(fixed)
    if patched is not None:
        try:
            result = json.loads(patched)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _close_open_structures(text: str) -> str | None:
    """Close a truncated JSON document, respecting actual nesting order.

    Counting braces is not enough: ``{"a": [{"b": 1`` must be closed as ``}]}``
    and not ``]}}``. This walks the text tracking a real stack, skipping over
    string contents so that braces inside strings are ignored.
    """
    stack: list[str] = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack and stack[-1] == ("{" if char == "}" else "["):
                stack.pop()
            else:
                return None  # genuinely malformed, not merely truncated

    if not stack and not in_string:
        return None  # nothing to close - the problem is something else

    patched = text
    if in_string:
        patched += '"'          # terminate the dangling string

    # Drop a trailing separator or dangling key so the result stays valid.
    patched = patched.rstrip()
    while patched and patched[-1] in ",:":
        patched = patched[:-1].rstrip()

    for opener in reversed(stack):
        patched += "}" if opener == "{" else "]"
    return patched


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
def _provider_message(response: httpx.Response) -> str:
    """The provider's own error text, if it sent one. Never includes the key."""
    try:
        payload = response.json()
    except Exception:
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")[:300]
    if isinstance(error, str):
        return error[:300]
    return str(payload.get("message") or "")[:300]


class LLMClient:
    """Async JSON-only LLM client."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # Set when the provider returns an error that retrying cannot fix
        # (bad key, no credits, unknown model). Once tripped, the client stops
        # calling out so an analysis does not waste N+2 doomed round-trips.
        self.disabled_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.settings.llm_configured and self.disabled_reason is None

    @property
    def degraded(self) -> bool:
        """True when the LLM was configured but had to be switched off."""
        return self.settings.llm_configured and self.disabled_reason is not None

    def _disable(self, reason: str) -> LLMError:
        """Trip the circuit breaker and build the error to raise."""
        if self.disabled_reason is None:
            self.disabled_reason = reason
            logger.error("LLM disabled for this session: %s", reason)
        return LLMError(reason)

    async def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send one prompt and return parsed JSON.

        Retries on transport errors, rate limits and unparseable JSON.
        """
        if not self.available:
            raise LLMNotConfigured(
                "No LLM_API_KEY configured (or DEMO_MODE is enabled)."
            )

        settings = self.settings
        attempts = max(1, settings.llm_max_retries + 1)
        last_error: Exception | None = None

        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            for attempt in range(attempts):
                try:
                    raw = await self._request(client, system, user, max_tokens)
                    return extract_json(raw)
                except LLMNotConfigured:
                    raise
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = LLMError(f"Could not reach the LLM provider: {exc}")
                    logger.warning("LLM transport error (attempt %s): %s", attempt + 1, exc)
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status in (401, 403):
                        # Never retry, and never log the key itself.
                        raise self._disable(
                            "The LLM API rejected the credentials (HTTP "
                            f"{status}). Check LLM_API_KEY in your .env file."
                        ) from exc
                    if status == 404:
                        raise self._disable(
                            f"The LLM endpoint returned 404. Check LLM_BASE_URL "
                            f"({settings.llm_base_url}) and LLM_MODEL "
                            f"({settings.llm_model})."
                        ) from exc
                    if status == 400:
                        # 400 is permanent for a given request shape. Surface the
                        # provider's own wording - it is usually actionable
                        # ("credit balance is too low", "model not found").
                        raise self._disable(
                            f"The LLM provider rejected the request (HTTP 400). "
                            f"{_provider_message(exc.response)}"
                        ) from exc
                    if status in (429, 500, 502, 503, 529):
                        last_error = LLMError(
                            f"The LLM provider is busy or unavailable (HTTP {status})."
                        )
                        logger.warning("LLM HTTP %s (attempt %s)", status, attempt + 1)
                    else:
                        raise LLMError(
                            f"The LLM API returned HTTP {status}."
                        ) from exc
                except LLMError as exc:
                    last_error = exc
                    logger.warning("LLM JSON error (attempt %s): %s", attempt + 1, exc)
                    # Nudge the model harder on the retry.
                    user = (
                        user
                        + "\n\nIMPORTANT: your previous response was not valid JSON. "
                        "Return ONLY a single valid JSON object, with no code fences "
                        "and no explanatory text."
                    )

                if attempt < attempts - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))

        raise last_error or LLMError("The LLM request failed.")

    async def _request(
        self,
        client: httpx.AsyncClient,
        system: str,
        user: str,
        max_tokens: int | None,
    ) -> str:
        settings = self.settings
        tokens = max_tokens or settings.llm_max_tokens

        if settings.llm_provider == "anthropic":
            url = f"{settings.llm_base_url}/messages"
            headers = {
                "x-api-key": settings.llm_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload: dict[str, Any] = {
                "model": settings.llm_model,
                "max_tokens": tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        else:
            url = f"{settings.llm_base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {settings.llm_api_key}",
                "content-type": "application/json",
            }
            payload = {
                "model": settings.llm_model,
                "max_tokens": tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
            }

        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return self._extract_text(data)

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Pull the assistant text out of either wire format."""
        # Anthropic: {"content": [{"type": "text", "text": "..."}]}
        content = data.get("content")
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if parts:
                return "\n".join(parts)

        # OpenAI: {"choices": [{"message": {"content": "..."}}]}
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") or {}
            text = message.get("content")
            if isinstance(text, str):
                return text
            # Some gateways return content as a list of parts.
            if isinstance(text, list):
                return "".join(
                    part.get("text", "") for part in text if isinstance(part, dict)
                )

        raise LLMError("The LLM response had an unrecognised structure.")
