"""LLM client behaviour, with the HTTP layer mocked.

No test here touches a real API. These cover the paths that keep a live demo
alive: malformed JSON, truncated JSON, auth failures, timeouts, and the
deterministic fallback when the model is unavailable.
"""

from __future__ import annotations

import httpx
import pytest

from app.ai.client import LLMClient, LLMError, LLMNotConfigured, extract_json
from app.analysis.analyzer import (
    _jd_from_llm,
    _profile_from_llm,
    enrich_with_narrative,
    extract_job_description,
    extract_resume_profile,
)
from app.config import Settings
from app.models.models import SkillImportance


def llm_settings(**overrides) -> Settings:
    """Settings with a (fake) key so the client considers itself available."""
    base = dict(
        llm_api_key="test-key-not-real",
        llm_model="test-model",
        llm_provider="anthropic",
        llm_base_url="https://example.invalid/v1",
        demo_mode=False,
        llm_max_retries=1,
    )
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------- #
# JSON extraction and repair
# --------------------------------------------------------------------------- #
class TestJsonExtraction:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_json_in_a_code_fence(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_in_a_bare_fence(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_prose_around_it(self):
        raw = 'Sure! Here is the result:\n{"a": 1}\nHope that helps.'
        assert extract_json(raw) == {"a": 1}

    def test_trailing_commas_are_repaired(self):
        assert extract_json('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}

    def test_comments_are_stripped(self):
        assert extract_json('{\n// a note\n"a": 1\n}') == {"a": 1}

    def test_truncated_json_is_closed(self):
        result = extract_json('{"analyses": [{"candidate_name": "A"')
        assert "analyses" in result

    def test_a_bare_list_is_wrapped(self):
        assert extract_json('[{"candidate_name": "A"}]') == {
            "analyses": [{"candidate_name": "A"}]
        }

    @pytest.mark.parametrize("raw", ["", "   ", "no json at all", "<html>nope</html>"])
    def test_unusable_responses_raise(self, raw):
        with pytest.raises(LLMError):
            extract_json(raw)


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
class TestAvailability:
    def test_demo_mode_disables_the_client(self):
        client = LLMClient(llm_settings(demo_mode=True))
        assert not client.available

    def test_missing_key_disables_the_client(self):
        client = LLMClient(llm_settings(llm_api_key=""))
        assert not client.available

    def test_a_configured_client_is_available(self):
        assert LLMClient(llm_settings()).available

    async def test_calling_an_unavailable_client_raises(self):
        client = LLMClient(llm_settings(demo_mode=True))
        with pytest.raises(LLMNotConfigured):
            await client.complete_json("system", "user")

    def test_settings_summary_never_leaks_the_key(self):
        summary = llm_settings(llm_api_key="super-secret-value").safe_summary()
        assert "super-secret-value" not in str(summary)
        assert summary["llm_api_key"] == "set"


# --------------------------------------------------------------------------- #
# Wire formats
# --------------------------------------------------------------------------- #
def mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestWireFormats:
    async def test_anthropic_response_is_parsed(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/messages")
            assert "x-api-key" in request.headers
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": '{"job_title": "Dev"}'}]},
            )

        client = LLMClient(llm_settings(llm_provider="anthropic"))
        _patch_httpx(monkeypatch, handler)

        assert await client.complete_json("s", "u") == {"job_title": "Dev"}

    async def test_openai_response_is_parsed(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/chat/completions")
            assert request.headers["authorization"].startswith("Bearer ")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"job_title": "Dev"}'}}]},
            )

        client = LLMClient(llm_settings(llm_provider="openai"))
        _patch_httpx(monkeypatch, handler)

        assert await client.complete_json("s", "u") == {"job_title": "Dev"}

    async def test_an_unrecognised_shape_raises(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, json={"unexpected": True})

        client = LLMClient(llm_settings())
        _patch_httpx(monkeypatch, handler)

        with pytest.raises(LLMError):
            await client.complete_json("s", "u")


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
class TestFailureHandling:
    async def test_invalid_api_key_gives_a_clear_message(self, monkeypatch):
        def handler(request):
            return httpx.Response(401, json={"error": "unauthorized"})

        client = LLMClient(llm_settings())
        _patch_httpx(monkeypatch, handler)

        with pytest.raises(LLMError) as exc:
            await client.complete_json("s", "u")
        assert "LLM_API_KEY" in str(exc.value)

    async def test_404_mentions_the_base_url_and_model(self, monkeypatch):
        def handler(request):
            return httpx.Response(404)

        client = LLMClient(llm_settings())
        _patch_httpx(monkeypatch, handler)

        with pytest.raises(LLMError) as exc:
            await client.complete_json("s", "u")
        assert "LLM_BASE_URL" in str(exc.value)

    async def test_rate_limit_is_retried_then_reported(self, monkeypatch):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(429)

        client = LLMClient(llm_settings(llm_max_retries=1))
        _patch_httpx(monkeypatch, handler)

        with pytest.raises(LLMError):
            await client.complete_json("s", "u")
        assert calls["n"] == 2, "Rate limits should be retried"

    async def test_bad_json_is_retried_and_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            text = "not json" if calls["n"] == 1 else '{"job_title": "Dev"}'
            return httpx.Response(200, json={"content": [{"type": "text", "text": text}]})

        client = LLMClient(llm_settings(llm_max_retries=2))
        _patch_httpx(monkeypatch, handler)

        assert await client.complete_json("s", "u") == {"job_title": "Dev"}
        assert calls["n"] == 2

    async def test_timeouts_are_retried_then_reported(self, monkeypatch):
        def handler(request):
            raise httpx.TimeoutException("timed out")

        client = LLMClient(llm_settings(llm_max_retries=1))
        _patch_httpx(monkeypatch, handler)

        with pytest.raises(LLMError) as exc:
            await client.complete_json("s", "u")
        assert "reach" in str(exc.value).lower()


def _patch_httpx(monkeypatch, handler) -> None:
    """Route every httpx.AsyncClient through a MockTransport."""
    original = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = mock_transport(handler)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


# --------------------------------------------------------------------------- #
# Validation of LLM payloads into Pydantic models
# --------------------------------------------------------------------------- #
class TestPayloadValidation:
    def test_jd_payload_is_validated(self, parse_fixture):
        document = parse_fixture("software_jd.txt")
        jd = _jd_from_llm(
            {
                "job_title": "Senior Backend Engineer",
                "required_skills": [{"skill": "Python", "context": "required"}],
                "preferred_skills": ["Kubernetes"],
                "min_years_experience": "5",
                "responsibilities": ["Build APIs"],
                "keywords": ["python", "api"],
            },
            document,
        )

        assert jd.job_title == "Senior Backend Engineer"
        assert jd.required_skills[0].importance == SkillImportance.REQUIRED
        assert jd.preferred_skills[0].skill == "Kubernetes"
        assert jd.min_years_experience == 5.0
        assert jd.raw_text == document.text

    def test_jd_payload_survives_junk_fields(self, parse_fixture):
        jd = _jd_from_llm(
            {
                "job_title": None,
                "required_skills": ["Python", 42, {"no_skill_key": 1}, {"skill": ""}],
                "min_years_experience": "not a number",
                "unexpected_field": "ignored",
            },
            parse_fixture("software_jd.txt"),
        )
        assert jd.job_title == "Unspecified Role"
        assert [r.skill for r in jd.required_skills] == ["Python"]
        assert jd.min_years_experience is None

    def test_duplicate_skills_are_dropped(self, parse_fixture):
        jd = _jd_from_llm(
            {"required_skills": ["Python", "python3", "PY"]},
            parse_fixture("software_jd.txt"),
        )
        assert len(jd.required_skills) == 1

    def test_resume_payload_is_validated(self, parse_fixture):
        document = parse_fixture("strong_software_resume.txt")
        profile = _profile_from_llm(
            {
                "candidate_name": "Priya Raghavan",
                "technical_skills": ["Python", "Docker"],
                "years_of_experience": 7,
                "education": [{"degree": "BTech", "field_of_study": "CS"}],
            },
            document,
        )
        assert profile.candidate_name == "Priya Raghavan"
        assert profile.years_of_experience == 7.0
        assert profile.raw_text == document.text
        assert profile.detected_domain

    def test_a_broken_resume_payload_falls_back_to_heuristics(self, parse_fixture):
        document = parse_fixture("strong_software_resume.txt")
        profile = _profile_from_llm({"education": "not a list"}, document)
        # Fallback still finds the candidate from the raw text.
        assert profile.candidate_name == "Priya Raghavan"


# --------------------------------------------------------------------------- #
# Fallback to the deterministic engine
# --------------------------------------------------------------------------- #
class TestFallback:
    async def test_jd_extraction_falls_back_when_the_llm_fails(
        self, parse_fixture, monkeypatch
    ):
        def handler(request):
            return httpx.Response(500)

        _patch_httpx(monkeypatch, handler)
        client = LLMClient(llm_settings(llm_max_retries=0))

        jd, llm_used = await extract_job_description(parse_fixture("software_jd.txt"), client)

        assert not llm_used
        assert jd.required_skills, "Fallback extraction must still produce requirements"

    async def test_resume_extraction_falls_back_when_the_llm_fails(
        self, parse_fixture, monkeypatch
    ):
        def handler(request):
            raise httpx.ConnectError("no network")

        _patch_httpx(monkeypatch, handler)
        client = LLMClient(llm_settings(llm_max_retries=0))

        profile, llm_used = await extract_resume_profile(
            parse_fixture("strong_software_resume.txt"), client
        )

        assert not llm_used
        assert profile.candidate_name == "Priya Raghavan"

    async def test_an_empty_llm_jd_result_falls_back(self, parse_fixture, monkeypatch):
        """A syntactically valid but useless response must not be trusted."""

        def handler(request):
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": '{"job_title": "Dev"}'}]},
            )

        _patch_httpx(monkeypatch, handler)
        client = LLMClient(llm_settings())

        jd, llm_used = await extract_job_description(parse_fixture("software_jd.txt"), client)

        assert not llm_used
        assert jd.required_skills

    async def test_demo_mode_never_calls_the_api(self, parse_fixture, monkeypatch):
        def handler(request):
            raise AssertionError("The LLM must not be called in demo mode")

        _patch_httpx(monkeypatch, handler)
        client = LLMClient(llm_settings(demo_mode=True))

        jd, llm_used = await extract_job_description(parse_fixture("software_jd.txt"), client)
        assert not llm_used
        assert jd.job_title


# --------------------------------------------------------------------------- #
# Narrative enrichment
# --------------------------------------------------------------------------- #
class TestNarrative:
    async def test_narrative_updates_text_but_never_scores(
        self, parse_fixture, software_jd, strong_resume, monkeypatch
    ):
        from app.analysis import skill_matcher
        from app.analysis.analyzer import analyze_one

        document = parse_fixture("strong_software_resume.txt")
        result = analyze_one(software_jd, strong_resume, document)
        original_ats = result.ats_score
        original_alignment = result.alignment_score

        payload = {
            "analyses": [
                {
                    "candidate_name": "Priya Raghavan",
                    "alignment_explanation": "A very strong fit for this backend role.",
                    "strong_points": ["Deep Python and AWS experience."],
                    "weak_points": ["No evidence of Kafka was found in the resume."],
                    "ats_explanation": "Scored highly on required skills.",
                    "extra_missing_areas": [
                        {
                            "area": "Production on-call leadership",
                            "category": "experience",
                            "priority": "medium",
                            "detail": "The JD expects mentoring; limited evidence found.",
                        }
                    ],
                }
            ]
        }

        def handler(request):
            import json

            return httpx.Response(
                200, json={"content": [{"type": "text", "text": json.dumps(payload)}]}
            )

        _patch_httpx(monkeypatch, handler)
        client = LLMClient(llm_settings())

        matches = skill_matcher.match_all(software_jd.all_requirements(), strong_resume)
        applied = await enrich_with_narrative(
            software_jd, [result], [strong_resume], [matches], client
        )

        assert applied
        assert result.alignment_explanation == "A very strong fit for this backend role."
        assert result.strong_points == ["Deep Python and AWS experience."]
        assert "Scored highly" in result.ats_explanation
        assert any(
            a.area == "Production on-call leadership" for a in result.missing_areas
        )
        # The deterministic numbers must be untouched.
        assert result.ats_score == original_ats
        assert result.alignment_score == original_alignment

    async def test_a_failed_narrative_keeps_the_deterministic_text(
        self, parse_fixture, software_jd, strong_resume, monkeypatch
    ):
        from app.analysis import skill_matcher
        from app.analysis.analyzer import analyze_one

        document = parse_fixture("strong_software_resume.txt")
        result = analyze_one(software_jd, strong_resume, document)
        original = result.alignment_explanation

        def handler(request):
            return httpx.Response(503)

        _patch_httpx(monkeypatch, handler)
        client = LLMClient(llm_settings(llm_max_retries=0))

        matches = skill_matcher.match_all(software_jd.all_requirements(), strong_resume)
        applied = await enrich_with_narrative(
            software_jd, [result], [strong_resume], [matches], client
        )

        assert not applied
        assert result.alignment_explanation == original
