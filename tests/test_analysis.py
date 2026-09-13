"""End-to-end analysis: the five required outputs, ranking, and confidence."""

from __future__ import annotations

import pytest

from app.analysis import recommender
from app.analysis.analyzer import analyze_one, run_analysis
from app.formatting import discord_format
from app.models.models import Confidence, Priority


async def analyse(jd_doc, resume_docs):
    return await run_analysis(jd_doc, resume_docs)


# --------------------------------------------------------------------------- #
# The five required outputs
# --------------------------------------------------------------------------- #
class TestRequiredOutputs:
    @pytest.fixture
    def result(self, parse_fixture, software_jd, strong_resume):
        document = parse_fixture("strong_software_resume.txt")
        return analyze_one(software_jd, strong_resume, document)

    def test_output_1_ats_score_and_breakdown(self, result):
        assert 0 <= result.ats_score <= 100
        breakdown = result.ats_breakdown.as_dict()
        assert len(breakdown) == 6
        assert all(0 <= v <= 100 for v in breakdown.values())
        assert result.ats_explanation

    def test_output_2_skill_set_match(self, result):
        assert 0 <= result.skill_match_score <= 100
        assert result.matched_skills, "A strong resume should have matched skills"
        # Buckets must be disjoint.
        names = (
            [m.skill for m in result.matched_skills]
            + [m.skill for m in result.partial_skills]
            + [m.skill for m in result.missing_skills]
        )
        assert len(names) == len(set(names))

    def test_output_3_missing_areas_are_prioritised_and_categorised(
        self, parse_fixture, software_jd, weak_resume
    ):
        document = parse_fixture("weak_software_resume.txt")
        result = analyze_one(software_jd, weak_resume, document)

        assert result.missing_areas
        assert any(a.priority == Priority.HIGH for a in result.missing_areas)
        assert all(a.detail for a in result.missing_areas)
        categories = {a.category for a in result.missing_areas}
        assert categories - {"technical_skill"}, "Gaps beyond keywords should be found"

    def test_output_4_alignment(self, result):
        assert 0 <= result.alignment_score <= 100
        assert len(result.alignment_breakdown.as_dict()) == 6
        assert result.alignment_explanation
        assert result.strong_points
        assert result.weak_points

    def test_output_5_courses_come_only_from_real_gaps(
        self, parse_fixture, software_jd, weak_resume
    ):
        document = parse_fixture("weak_software_resume.txt")
        result = analyze_one(software_jd, weak_resume, document)

        assert result.recommended_courses
        missing_and_partial = {
            m.skill.lower() for m in result.missing_skills + result.partial_skills
        }
        for course in result.recommended_courses:
            assert course.reason, "Every course must justify itself"
            assert course.skill.lower() in missing_and_partial, (
                f"Course for '{course.skill}' does not correspond to a real gap"
            )

    def test_no_courses_when_there_are_no_gaps(self, software_jd, parse_fixture):
        """A candidate covering everything should be told no course is needed."""
        from app.analysis.heuristic_extractor import extract_resume

        document = parse_fixture("strong_software_resume.txt")
        profile = extract_resume(document)
        result = analyze_one(software_jd, profile, document)

        # This resume misses only preferred skills, so any course must be low priority.
        assert all(c.priority == Priority.LOW for c in result.recommended_courses)

    def test_every_course_has_a_link(self, parse_fixture, software_jd, weak_resume):
        document = parse_fixture("weak_software_resume.txt")
        result = analyze_one(software_jd, weak_resume, document)

        assert result.recommended_courses
        for course in result.recommended_courses:
            assert course.url.startswith("https://"), (
                f"Course '{course.title}' has no usable link"
            )

    def test_course_links_are_curated_or_search_urls_never_invented(
        self, parse_fixture, software_jd, weak_resume
    ):
        """Links must be a curated page or a provider search - no fabricated deep links."""
        from urllib.parse import urlparse

        from app.analysis.recommender import COURSE_URLS, course_url

        document = parse_fixture("weak_software_resume.txt")
        result = analyze_one(software_jd, weak_resume, document)

        curated = set(COURSE_URLS.values())
        for course in result.recommended_courses:
            if course.url in curated:
                continue
            # Otherwise it must be a search URL (has a query string) or a
            # provider landing page - never an invented deep course path.
            parsed = urlparse(course.url)
            assert parsed.query or parsed.path in ("", "/"), (
                f"'{course.url}' looks like a fabricated deep link"
            )

    def test_links_point_at_known_providers(self):
        from urllib.parse import urlparse

        from app.analysis.recommender import CATALOG, COURSE_URLS, course_url

        allowed = {
            "coursera.org", "www.coursera.org", "udemy.com", "www.udemy.com",
            "edx.org", "www.edx.org", "kaggle.com", "www.kaggle.com",
            "udacity.com", "www.udacity.com", "linkedin.com", "www.linkedin.com",
            "learn.microsoft.com", "skillbuilder.aws", "cloudskillsboost.google",
            "www.cloudskillsboost.google", "docs.docker.com", "kodekloud.com",
            "docs.github.com", "developer.hashicorp.com", "www.jenkins.io",
            "training.linuxfoundation.org", "go.dev", "www.apollographql.com",
            "flask.palletsprojects.com", "fastapi.tiangolo.com", "spring.academy",
            "www.postgresql.org", "learn.mongodb.com", "www.elastic.co",
            "pytorch.org", "www.tensorflow.org", "scikit-learn.org", "numpy.org",
            "www.deeplearning.ai", "docs.pytest.org", "www.solidworks.com",
            "www.autodesk.com", "www.3ds.com", "www.ptc.com", "www.asme.org",
            "www.ansys.com", "nptel.ac.in", "asq.org", "matlabacademy.mathworks.com",
            "www.tableau.com", "www.freecodecamp.org",
        }
        for skill, (title, provider, _) in CATALOG.items():
            host = urlparse(course_url(skill, title, provider)).netloc
            assert host in allowed, f"{skill} -> unexpected host {host}"

    def test_curated_links_are_https(self):
        from app.analysis.recommender import COURSE_URLS

        for skill, url in COURSE_URLS.items():
            assert url.startswith("https://"), f"{skill} link is not https"

    def test_learning_roadmap_is_built_from_the_courses(
        self, parse_fixture, software_jd, weak_resume
    ):
        document = parse_fixture("weak_software_resume.txt")
        result = analyze_one(software_jd, weak_resume, document)

        assert result.learning_roadmap
        assert len(result.learning_roadmap) <= 4
        assert [i.priority for i in result.learning_roadmap] == list(
            range(1, len(result.learning_roadmap) + 1)
        )
        assert all(item.topics for item in result.learning_roadmap)


# --------------------------------------------------------------------------- #
# Evidence-based language
# --------------------------------------------------------------------------- #
class TestEvidenceBasedLanguage:
    def test_gaps_are_phrased_as_missing_evidence(
        self, parse_fixture, software_jd, weak_resume
    ):
        document = parse_fixture("weak_software_resume.txt")
        result = analyze_one(software_jd, weak_resume, document)

        for point in result.weak_points:
            lowered = point.lower()
            assert "cannot" not in lowered
            assert "does not know" not in lowered
        assert any("no evidence" in p.lower() for p in result.weak_points)

    def test_matched_skills_carry_evidence(self, parse_fixture, software_jd, strong_resume):
        document = parse_fixture("strong_software_resume.txt")
        result = analyze_one(software_jd, strong_resume, document)
        assert any(m.evidence for m in result.matched_skills)


# --------------------------------------------------------------------------- #
# Scenario 6: one JD, three resumes
# --------------------------------------------------------------------------- #
class TestMultipleResumes:
    @pytest.fixture
    async def report(self, parse_fixture):
        jd = parse_fixture("software_jd.txt")
        resumes = [
            parse_fixture("strong_software_resume.txt"),
            parse_fixture("weak_software_resume.txt"),
            parse_fixture("mechanical_resume.txt"),
        ]
        return await analyse(jd, resumes)

    async def test_all_three_resumes_are_analysed(self, report):
        assert len(report.results) == 3

    async def test_candidates_are_ranked_best_first(self, report):
        scores = [r.ats_score for r in report.ranked]
        assert scores == sorted(scores, reverse=True)

    async def test_the_best_candidate_is_the_strong_one(self, report):
        assert report.best.candidate_name == "Priya Raghavan"

    async def test_the_mechanical_resume_ranks_last_and_is_flagged(self, report):
        last = report.ranked[-1]
        assert last.candidate_name == "Rohit Deshmukh"
        assert last.role_mismatch.detected

    async def test_each_candidate_gets_a_full_independent_analysis(self, report):
        for result in report.results:
            assert result.candidate_name
            assert result.ats_breakdown.as_dict()
            assert result.alignment_explanation
            assert result.confidence in set(Confidence)

    async def test_a_single_resume_still_produces_a_report(self, parse_fixture):
        report = await analyse(
            parse_fixture("software_jd.txt"),
            [parse_fixture("strong_software_resume.txt")],
        )
        assert len(report.results) == 1
        assert report.best is not None

    async def test_progress_messages_are_emitted(self, parse_fixture):
        seen: list[str] = []

        async def progress(message: str) -> None:
            seen.append(message)

        await run_analysis(
            parse_fixture("software_jd.txt"),
            [parse_fixture("strong_software_resume.txt")],
            progress=progress,
        )
        assert len(seen) >= 3


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
class TestConfidence:
    def test_good_documents_give_high_confidence(
        self, parse_fixture, software_jd, strong_resume
    ):
        document = parse_fixture("strong_software_resume.txt")
        result = analyze_one(software_jd, strong_resume, document, llm_used=True)
        assert result.confidence == Confidence.HIGH

    def test_a_thin_resume_lowers_confidence(self, software_jd):
        from app.analysis.heuristic_extractor import extract_resume
        from app.parsers.text_parser import from_string

        document = from_string("John Smith\nPython developer.", "thin.txt")
        profile = extract_resume(document)
        result = analyze_one(software_jd, profile, document)

        assert result.confidence in {Confidence.LOW, Confidence.MEDIUM}
        assert result.confidence_reason

    def test_confidence_has_an_emoji_for_display(self):
        assert Confidence.HIGH.emoji
        assert Confidence.LOW.emoji != Confidence.HIGH.emoji


# --------------------------------------------------------------------------- #
# Discord formatting
# --------------------------------------------------------------------------- #
class TestDiscordFormatting:
    async def test_every_message_fits_discord_limit(self, parse_fixture):
        report = await analyse(
            parse_fixture("software_jd.txt"),
            [
                parse_fixture("strong_software_resume.txt"),
                parse_fixture("weak_software_resume.txt"),
                parse_fixture("mechanical_resume.txt"),
            ],
        )
        messages = discord_format.format_full_report(report)

        assert messages
        for message in messages:
            assert len(message) <= discord_format.DISCORD_LIMIT

    async def test_report_contains_all_five_sections(self, parse_fixture):
        report = await analyse(
            parse_fixture("software_jd.txt"),
            [parse_fixture("weak_software_resume.txt")],
        )
        blob = "\n".join(discord_format.format_full_report(report))

        assert "ATS SCORE" in blob
        assert "SKILL SET MATCH" in blob
        assert "MISSING / IMPROVEMENT AREAS" in blob
        assert "JOB ALIGNMENT" in blob
        assert "RECOMMENDED COURSES" in blob
        assert "LEARNING ROADMAP" in blob

    async def test_comparison_table_appears_for_multiple_resumes(self, parse_fixture):
        report = await analyse(
            parse_fixture("software_jd.txt"),
            [
                parse_fixture("strong_software_resume.txt"),
                parse_fixture("weak_software_resume.txt"),
            ],
        )
        blob = "\n".join(discord_format.format_full_report(report))
        assert "CANDIDATE COMPARISON" in blob
        assert "Best aligned candidate" in blob

    async def test_role_mismatch_banner_is_shown(self, parse_fixture):
        report = await analyse(
            parse_fixture("mechanical_jd.txt"),
            [parse_fixture("strong_software_resume.txt")],
        )
        blob = "\n".join(discord_format.format_full_report(report))
        assert "ROLE MISMATCH DETECTED" in blob

    def test_code_fences_stay_balanced_when_a_block_is_split(self):
        big_block = "```\n" + "\n".join(f"row {i:04d} of data" for i in range(400)) + "\n```"
        messages = discord_format.pack_messages([big_block])

        assert len(messages) > 1
        for message in messages:
            assert message.count("```") % 2 == 0, "Unbalanced code fence after split"
            assert len(message) <= discord_format.DISCORD_LIMIT

    def test_pack_messages_handles_an_unbroken_giant_line(self):
        messages = discord_format.pack_messages(["x" * 9000])
        assert all(len(m) <= discord_format.DISCORD_LIMIT for m in messages)
        assert sum(len(m) for m in messages) >= 9000

    def test_bar_rendering(self):
        assert discord_format.bar(100).count("█") == 12
        assert discord_format.bar(0).count("░") == 12
        assert len(discord_format.bar(55)) == 12

    def test_skill_names_are_displayed_properly(self, parse_fixture, software_jd,
                                                 strong_resume):
        document = parse_fixture("strong_software_resume.txt")
        result = analyze_one(software_jd, strong_resume, document)
        blob = "\n".join(discord_format.format_result(result))

        assert "AWS" in blob
        assert "CI/CD" in blob
        assert "aws" not in blob.replace("AWS", "")


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
class TestRobustness:
    async def test_empty_resume_does_not_crash_the_pipeline(self, parse_fixture):
        from app.parsers.text_parser import from_string

        report = await analyse(
            parse_fixture("software_jd.txt"),
            [from_string("", "empty.txt")],
        )
        assert len(report.results) == 1
        assert report.results[0].ats_score >= 0
        assert report.results[0].confidence == Confidence.LOW

    async def test_a_jd_with_no_clear_requirements_still_completes(self, parse_fixture):
        from app.parsers.text_parser import from_string

        report = await analyse(
            from_string("We want a great person to join our team.", "vague_jd.txt"),
            [parse_fixture("strong_software_resume.txt")],
        )
        assert len(report.results) == 1
        assert report.results[0].confidence in {Confidence.LOW, Confidence.MEDIUM}

    def test_improvement_suggestions_are_capped(
        self, parse_fixture, software_jd, weak_resume
    ):
        document = parse_fixture("weak_software_resume.txt")
        result = analyze_one(software_jd, weak_resume, document)
        assert len(result.improvement_suggestions) <= 3


# --------------------------------------------------------------------------- #
# Regression: skill match must not suppress alignment gaps
# --------------------------------------------------------------------------- #
class TestSkillMatchVsAlignmentConsistency:
    """A 100% skill match with weak alignment must still report gaps.

    Reproduces the reported inconsistency: 'Skill match 100%' plus 'No
    significant gaps were found', while Experience sat at 30%, Projects 35%,
    Keywords 44% and Education 50%.
    """

    @pytest.fixture
    def thin_but_matching(self):
        """A resume naming every required skill but evidencing almost nothing."""
        from app.analysis.heuristic_extractor import extract_resume
        from app.parsers.text_parser import from_string

        document = from_string(
            "Sam Taylor\n"
            "sam.taylor@example.com\n\n"
            "Technical Skills\n"
            "Python, PostgreSQL, Docker, AWS, REST APIs, Git, CI/CD, "
            "microservices, SQL, unit testing\n\n"
            "Work Experience\n"
            "Associate, Contoso, 2023 - 2024\n"
            "- Helped the team with assorted tasks\n",
            "thin_matching.txt",
        )
        return extract_resume(document), document

    def _analyse(self, jd, pair):
        profile, document = pair
        return analyze_one(jd, profile, document)

    def test_setup_reproduces_the_reported_shape(self, software_jd, thin_but_matching):
        """Guard the premise: high skill match, low alignment dimensions."""
        result = self._analyse(software_jd, thin_but_matching)
        assert result.skill_match_score >= 80
        weak = [
            name
            for name, score in result.alignment_breakdown.as_dict().items()
            if score < 75
        ]
        assert weak, "Fixture no longer produces weak alignment dimensions"

    def test_missing_areas_are_not_empty_when_alignment_is_weak(
        self, software_jd, thin_but_matching
    ):
        result = self._analyse(software_jd, thin_but_matching)
        assert result.missing_areas, (
            "Alignment scored poorly but no gaps were reported"
        )

    def test_weak_alignment_dimensions_become_gaps(
        self, software_jd, thin_but_matching
    ):
        from app.analysis.analyzer import weak_alignment_dimensions

        result = self._analyse(software_jd, thin_but_matching)
        weak = {name for name, _ in weak_alignment_dimensions(result.alignment_breakdown)}
        categories = {a.category for a in result.missing_areas}

        mapping = {
            "Experience": "experience",
            "Projects": "project_evidence",
            "Responsibilities": "responsibility",
            "Keywords": "keywords",
            "Education": "education",
        }
        for dimension, category in mapping.items():
            if dimension in weak:
                assert category in categories, (
                    f"{dimension} scored below threshold but produced no gap"
                )

    def test_weak_points_never_claim_no_gaps_while_alignment_is_low(
        self, software_jd, thin_but_matching
    ):
        result = self._analyse(software_jd, thin_but_matching)
        blob = " ".join(result.weak_points).lower()
        assert "no material gaps" not in blob
        assert result.weak_points

    def test_rendered_output_is_self_consistent(self, software_jd, thin_but_matching):
        """The rendered message must not contradict the numbers it prints."""
        result = self._analyse(software_jd, thin_but_matching)
        blob = "\n".join(discord_format.format_result(result))

        assert "No significant gaps were found" not in blob
        assert "KEY ALIGNMENT GAPS" in blob
        assert "alignment is low" in blob

    def test_gaps_cite_real_evidence(self, software_jd, thin_but_matching):
        """Every alignment gap must explain itself, not just assert a number."""
        result = self._analyse(software_jd, thin_but_matching)
        for area in result.missing_areas:
            assert area.detail, f"Gap '{area.area}' has no justification"
            assert len(area.detail) > 25

    def test_courses_follow_the_knowledge_gaps(self, software_jd, thin_but_matching):
        result = self._analyse(software_jd, thin_but_matching)
        for course in result.recommended_courses:
            assert course.reason

    def test_no_gaps_are_invented_for_a_genuinely_strong_resume(
        self, parse_fixture, software_jd, strong_resume
    ):
        """The fix must not manufacture gaps where alignment is actually good."""
        document = parse_fixture("strong_software_resume.txt")
        result = analyze_one(software_jd, strong_resume, document)

        from app.analysis.analyzer import weak_alignment_dimensions

        weak = {name for name, _ in weak_alignment_dimensions(result.alignment_breakdown)}
        alignment_categories = {"experience", "project_evidence", "keywords", "education"}
        raised = {a.category for a in result.missing_areas} & alignment_categories

        for category in raised:
            dimension = {
                "experience": "Experience",
                "project_evidence": "Projects",
                "keywords": "Keywords",
                "education": "Education",
            }[category]
            assert dimension in weak, (
                f"Gap raised for {dimension} even though it scored well"
            )

    def test_keyword_gaps_only_name_real_skills(self, software_jd, thin_but_matching):
        """Generic JD prose must never become a course recommendation."""
        from app.analysis.recommender import keyword_learning_gaps

        profile, _ = thin_but_matching
        result = self._analyse(software_jd, thin_but_matching)
        gaps = keyword_learning_gaps(software_jd, profile, result.alignment_breakdown)

        from app.analysis import skill_taxonomy as tax
        from app.analysis.recommender import CATALOG

        for skill, _, _ in gaps:
            canon = tax.canonical(skill)
            assert canon in tax.ALIASES or canon in CATALOG
