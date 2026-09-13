"""Semantic skill matching: synonyms, related skills, and required vs preferred."""

from __future__ import annotations

import pytest

from app.analysis import skill_matcher, skill_taxonomy as tax
from app.models.models import (
    JDRequirement,
    MatchStatus,
    ResumeProfile,
    SkillImportance,
)


def make_profile(skills: list[str], text: str = "") -> ResumeProfile:
    """A minimal profile carrying an explicit skill list."""
    return ResumeProfile(
        candidate_name="Test Candidate",
        technical_skills=skills,
        raw_text=text or " ".join(skills),
    )


def require(skill: str, importance=SkillImportance.REQUIRED) -> JDRequirement:
    return JDRequirement(skill=skill, importance=importance)


# --------------------------------------------------------------------------- #
# Canonicalisation / synonyms
# --------------------------------------------------------------------------- #
class TestSynonyms:
    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("JS", "javascript"),
            ("JavaScript", "javascript"),
            ("Postgres", "postgresql"),
            ("PostgreSQL", "postgresql"),
            ("Amazon Web Services", "aws"),
            ("AWS", "aws"),
            ("RESTful APIs", "rest api"),
            ("REST APIs", "rest api"),
            ("ML", "machine learning"),
            ("NLP", "natural language processing"),
            ("React.js", "react"),
            ("ReactJS", "react"),
            ("K8s", "kubernetes"),
            ("CI/CD", "ci/cd"),
            ("Continuous Integration", "ci/cd"),
            ("FEA", "finite element analysis"),
            ("GDT", "gd&t"),
        ],
    )
    def test_aliases_map_to_the_same_canonical_skill(self, alias, expected):
        assert tax.canonical(alias) == expected

    def test_synonym_in_the_resume_fully_satisfies_the_jd(self):
        """A resume saying 'JS' must fully match a JD asking for 'JavaScript'."""
        profile = make_profile(["JS", "Postgres", "AWS"])
        results = skill_matcher.match_all(
            [require("JavaScript"), require("PostgreSQL"), require("Amazon Web Services")],
            profile,
        )
        assert all(r.status == MatchStatus.EXACT for r in results)

    def test_noise_words_are_stripped(self):
        assert tax.canonical("strong experience with Docker") == "docker"
        assert tax.canonical("3+ years of Python") == "python"


# --------------------------------------------------------------------------- #
# Exact / partial / missing
# --------------------------------------------------------------------------- #
class TestMatchStatuses:
    def test_exact_match_from_the_skill_list(self):
        profile = make_profile(["Python", "Docker"])
        result = skill_matcher.match_skill(require("Docker"), profile)
        assert result.status == MatchStatus.EXACT
        assert result.matched_via.lower() == "docker"

    def test_exact_match_found_in_the_resume_body(self):
        """A skill in prose counts even if the extractor missed it."""
        profile = make_profile(
            [], text="Built and deployed containerised services using Docker daily."
        )
        result = skill_matcher.match_skill(require("Docker"), profile)
        assert result.status == MatchStatus.EXACT
        assert result.evidence

    def test_missing_skill_is_reported_as_missing(self):
        profile = make_profile(["Python", "Django"])
        result = skill_matcher.match_skill(require("Kubernetes"), profile)
        assert result.status == MatchStatus.MISSING
        assert result.matched_via == ""

    def test_partial_match_for_a_related_database(self):
        """MySQL is related to PostgreSQL, but must not count as equivalent."""
        profile = make_profile(["MySQL"])
        result = skill_matcher.match_skill(require("PostgreSQL"), profile)

        assert result.status == MatchStatus.PARTIAL
        assert result.credit == 0.5
        assert "relational" in result.relation.lower()

    def test_partial_match_for_a_related_cloud_provider(self):
        profile = make_profile(["Azure"])
        result = skill_matcher.match_skill(require("AWS"), profile)
        assert result.status == MatchStatus.PARTIAL

    def test_partial_match_for_a_related_python_framework(self):
        profile = make_profile(["Django"])
        result = skill_matcher.match_skill(require("Flask"), profile)
        assert result.status == MatchStatus.PARTIAL
        assert "python web framework" in result.relation.lower()

    @pytest.mark.parametrize(
        "resume_skill,jd_skill",
        [
            ("Python", "Java"),      # different languages are NOT interchangeable
            ("Java", "Python"),
            ("Excel", "Kubernetes"),
            ("AutoCAD", "Python"),
            ("React", "Django"),
        ],
    )
    def test_unrelated_skills_are_never_conflated(self, resume_skill, jd_skill):
        profile = make_profile([resume_skill])
        result = skill_matcher.match_skill(require(jd_skill), profile)
        assert result.status == MatchStatus.MISSING

    def test_credit_values(self):
        profile_exact = make_profile(["Docker"])
        profile_partial = make_profile(["MySQL"])
        profile_missing = make_profile(["Excel"])

        assert skill_matcher.match_skill(require("Docker"), profile_exact).credit == 1.0
        assert skill_matcher.match_skill(require("PostgreSQL"), profile_partial).credit == 0.5
        assert skill_matcher.match_skill(require("Docker"), profile_missing).credit == 0.0


# --------------------------------------------------------------------------- #
# Required vs preferred weighting
# --------------------------------------------------------------------------- #
class TestRequiredVsPreferred:
    def test_required_skills_outweigh_preferred_skills(self):
        """Nice-to-haves must not compensate for a missing must-have."""
        profile = make_profile(["Kubernetes", "Terraform", "Kafka"])
        requirements = [
            require("Python"),                                    # required, missing
            require("Java"),                                      # required, missing
            require("Kubernetes", SkillImportance.PREFERRED),     # preferred, matched
            require("Terraform", SkillImportance.PREFERRED),      # preferred, matched
            require("Kafka", SkillImportance.PREFERRED),          # preferred, matched
        ]
        score = skill_matcher.skill_match_score(
            skill_matcher.match_all(requirements, profile)
        )
        # 3 preferred hits (weight 1) vs 2 required misses (weight 3): 3/9 = 33%
        assert score < 40

    def test_a_related_skill_still_earns_partial_credit_for_a_must_have(self):
        """Kubernetes on the resume partially covers a Docker requirement."""
        profile = make_profile(["Kubernetes"])
        result = skill_matcher.match_skill(require("Docker"), profile)
        assert result.status == MatchStatus.PARTIAL
        assert result.credit == 0.5

    def test_matching_the_required_skills_scores_highly(self):
        profile = make_profile(["Python", "Docker"])
        requirements = [
            require("Python"),
            require("Docker"),
            require("Kubernetes", SkillImportance.PREFERRED),
        ]
        score = skill_matcher.skill_match_score(
            skill_matcher.match_all(requirements, profile)
        )
        assert score > 80

    def test_perfect_coverage_scores_100(self):
        profile = make_profile(["Python", "Docker"])
        results = skill_matcher.match_all([require("Python"), require("Docker")], profile)
        assert skill_matcher.skill_match_score(results) == 100.0

    def test_no_requirements_scores_zero_not_a_crash(self):
        assert skill_matcher.skill_match_score([]) == 0.0

    def test_split_by_status_puts_required_first(self):
        profile = make_profile(["Python"])
        requirements = [
            require("Redis", SkillImportance.PREFERRED),
            require("Docker"),
            require("Python"),
        ]
        exact, partial, missing = skill_matcher.split_by_status(
            skill_matcher.match_all(requirements, profile)
        )
        assert [m.skill for m in exact] == ["Python"]
        assert partial == []
        assert missing[0].importance == SkillImportance.REQUIRED


# --------------------------------------------------------------------------- #
# Deduplication and keyword coverage
# --------------------------------------------------------------------------- #
class TestMatchingBehaviour:
    def test_duplicate_requirements_are_matched_once(self):
        profile = make_profile(["Python"])
        results = skill_matcher.match_all(
            [require("Python"), require("python3"), require("Py")], profile
        )
        assert len(results) == 1

    def test_keyword_coverage_counts_synonyms(self):
        profile = make_profile(["JS", "Postgres"])
        coverage = skill_matcher.keyword_coverage(
            ["JavaScript", "PostgreSQL"], profile
        )
        assert coverage == 100.0

    def test_keyword_coverage_with_no_keywords(self):
        assert skill_matcher.keyword_coverage([], make_profile(["Python"])) == 0.0

    def test_evidence_snippet_is_captured_from_the_resume(self, strong_resume):
        result = skill_matcher.match_skill(require("Docker"), strong_resume)
        assert result.status == MatchStatus.EXACT
        assert "docker" in result.evidence.lower()

    def test_short_skill_names_do_not_match_inside_words(self):
        """'Go' must not match inside 'Google' or 'algorithm'."""
        profile = make_profile([], text="Worked at Google on algorithms and goals.")
        result = skill_matcher.match_skill(require("Go"), profile)
        assert result.status == MatchStatus.MISSING


# --------------------------------------------------------------------------- #
# Domain classification (feeds role-mismatch detection)
# --------------------------------------------------------------------------- #
class TestDomainClassification:
    def test_software_resume_classifies_as_software(self, strong_resume):
        assert strong_resume.detected_domain == "software_engineering"

    def test_mechanical_resume_classifies_as_mechanical(self, mechanical_resume):
        assert mechanical_resume.detected_domain == "mechanical_engineering"

    def test_software_and_data_domains_are_compatible(self):
        assert tax.domains_compatible("software_engineering", "data_science_ml")

    def test_software_and_mechanical_are_incompatible(self):
        assert not tax.domains_compatible(
            "software_engineering", "mechanical_engineering"
        )

    def test_unknown_domain_never_triggers_incompatibility(self):
        """Too little signal must not be treated as a mismatch."""
        assert tax.domains_compatible("unknown", "mechanical_engineering")

    def test_sparse_text_stays_unknown(self):
        domain, _ = tax.classify_domain("Hello there.")
        assert domain == "unknown"
