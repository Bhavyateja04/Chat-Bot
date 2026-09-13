"""ATS scoring engine, alignment scoring and role-mismatch detection."""

from __future__ import annotations

import pytest

from app.analysis import ats_scorer, role_matcher, skill_matcher
from app.models.models import (
    ATSBreakdown,
    EducationEntry,
    ExperienceEntry,
    JDRequirement,
    JobDescription,
    Priority,
    ResumeProfile,
    RoleMismatch,
    SkillImportance,
)


def score_pair(jd, profile):
    """Run the full scoring pipeline for a JD/resume pair."""
    matches = skill_matcher.match_all(jd.all_requirements(), profile)
    mismatch = role_matcher.detect(jd, profile, matches)
    ats, breakdown = ats_scorer.compute_ats_score(jd, profile, matches, mismatch)
    return ats, breakdown, mismatch, matches


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
class TestWeights:
    def test_weights_sum_to_one(self):
        assert sum(ats_scorer.WEIGHTS.values()) == pytest.approx(1.0)

    def test_weighted_total_of_a_perfect_breakdown_is_100(self):
        perfect = ATSBreakdown(
            keyword_match=100, required_skills=100, experience=100,
            projects=100, education=100, resume_structure=100,
        )
        assert ats_scorer.weighted_total(perfect) == pytest.approx(100.0)

    def test_weighted_total_of_an_empty_breakdown_is_zero(self):
        assert ats_scorer.weighted_total(ATSBreakdown()) == 0.0

    def test_required_skills_carry_the_largest_single_weight(self):
        assert ats_scorer.WEIGHTS["required_skills"] == max(ats_scorer.WEIGHTS.values())

    def test_every_component_is_in_range(self, software_jd, strong_resume):
        _, breakdown, _, _ = score_pair(software_jd, strong_resume)
        for name, value in breakdown.as_dict().items():
            assert 0.0 <= value <= 100.0, f"{name} out of range: {value}"


# --------------------------------------------------------------------------- #
# Scenario 1 & 2: strong and weak software resumes against a software JD
# --------------------------------------------------------------------------- #
class TestSoftwareScenarios:
    def test_strong_sde_resume_scores_highly(self, software_jd, strong_resume):
        ats, breakdown, mismatch, matches = score_pair(software_jd, strong_resume)

        assert ats >= 75, f"Strong candidate under-scored: {ats}"
        assert not mismatch.detected
        assert breakdown.required_skills >= 85
        assert skill_matcher.skill_match_score(matches) >= 80

    def test_weak_sde_resume_scores_poorly(self, software_jd, weak_resume):
        ats, breakdown, mismatch, matches = score_pair(software_jd, weak_resume)

        assert ats <= 50, f"Weak candidate over-scored: {ats}"
        assert breakdown.required_skills <= 40
        # Same field, so this is a skills gap and not a role mismatch.
        assert not mismatch.detected

    def test_strong_resume_beats_weak_resume(
        self, software_jd, strong_resume, weak_resume
    ):
        strong_score, _, _, _ = score_pair(software_jd, strong_resume)
        weak_score, _, _, _ = score_pair(software_jd, weak_resume)
        assert strong_score > weak_score + 25


# --------------------------------------------------------------------------- #
# Scenario 3: the recruiter's headline test - mechanical JD vs software resume
# --------------------------------------------------------------------------- #
class TestRoleMismatch:
    def test_mechanical_jd_against_software_resume_is_flagged(
        self, mechanical_jd, strong_resume
    ):
        ats, _, mismatch, _ = score_pair(mechanical_jd, strong_resume)

        assert mismatch.detected, "Role mismatch was not detected"
        assert mismatch.severity == Priority.HIGH
        assert mismatch.jd_domain == "mechanical_engineering"
        assert mismatch.resume_domain == "software_engineering"
        assert ats <= 30, f"Mismatched resume scored far too high: {ats}"
        assert mismatch.reason
        assert mismatch.critical_missing

    def test_software_jd_against_mechanical_resume_is_flagged(
        self, software_jd, mechanical_resume
    ):
        ats, _, mismatch, _ = score_pair(software_jd, mechanical_resume)

        assert mismatch.detected
        assert ats <= 30
        assert any(
            "python" in s.lower() or "sql" in s.lower() or "docker" in s.lower()
            for s in mismatch.critical_missing
        )

    def test_a_keyword_dense_wrong_domain_resume_cannot_score_well(
        self, mechanical_jd, strong_resume
    ):
        """The whole point: lots of impressive but irrelevant keywords != a fit."""
        ats, breakdown, mismatch, _ = score_pair(mechanical_jd, strong_resume)

        # The resume genuinely parses well and has education - those components
        # can be high - yet the overall score must still be low.
        assert breakdown.resume_structure >= 60
        assert ats <= ats_scorer.MISMATCH_CEILING[Priority.HIGH]

    def test_matching_domain_is_never_flagged(self, mechanical_jd, mechanical_resume):
        ats, _, mismatch, _ = score_pair(mechanical_jd, mechanical_resume)
        assert not mismatch.detected
        assert ats >= 70, f"Correct-domain candidate under-scored: {ats}"

    def test_penalty_and_ceiling_are_applied(self):
        mismatch = RoleMismatch(detected=True, severity=Priority.HIGH)
        assert ats_scorer.apply_mismatch_penalty(90.0, mismatch) <= 30.0

    def test_no_penalty_without_a_mismatch(self):
        assert ats_scorer.apply_mismatch_penalty(90.0, RoleMismatch()) == 90.0

    def test_detection_is_silent_when_a_document_is_unclassifiable(self):
        """Too little signal must not produce a false mismatch accusation."""
        jd = JobDescription(job_title="Role", raw_text="We need someone good.")
        profile = ResumeProfile(candidate_name="X", raw_text="I am a person.")
        mismatch = role_matcher.detect(jd, profile, [])
        assert not mismatch.detected


# --------------------------------------------------------------------------- #
# Scenario 4 & 5: adjacent-role scenarios
# --------------------------------------------------------------------------- #
class TestAdjacentRoles:
    def test_ml_jd_against_a_software_resume_is_a_gap_not_a_mismatch(
        self, ml_jd, strong_resume
    ):
        """ML and software engineering are adjacent: score it down, don't flag it."""
        ats, _, mismatch, _ = score_pair(ml_jd, strong_resume)

        assert not mismatch.detected, "Adjacent domains must not be a role mismatch"
        assert ats < 70, f"Software resume should not look like an ML fit: {ats}"

    def test_backend_jd_against_a_fullstack_resume_scores_reasonably(
        self, backend_jd, fullstack_resume
    ):
        """A full-stack candidate is a genuine fit for a backend role."""
        ats, _, mismatch, matches = score_pair(backend_jd, fullstack_resume)

        assert not mismatch.detected
        assert ats >= 60, f"Full-stack candidate under-scored for backend role: {ats}"

    def test_fullstack_resume_shows_partial_database_credit(
        self, backend_jd, fullstack_resume
    ):
        """The JD wants PostgreSQL; the resume has MySQL - partial, not full."""
        matches = skill_matcher.match_all(backend_jd.all_requirements(), fullstack_resume)
        postgres = [m for m in matches if "postgres" in m.skill.lower()]
        assert postgres, "PostgreSQL requirement not found in the JD"
        assert postgres[0].status.value == "partial"


# --------------------------------------------------------------------------- #
# Individual components
# --------------------------------------------------------------------------- #
class TestComponents:
    def test_experience_score_rewards_meeting_the_requirement(self):
        jd = JobDescription(job_title="Dev", min_years_experience=3.0)
        senior = ResumeProfile(
            candidate_name="A", years_of_experience=6.0,
            experience=[ExperienceEntry(title="Engineer", description="built things")],
        )
        junior = ResumeProfile(
            candidate_name="B", years_of_experience=1.0,
            experience=[ExperienceEntry(title="Intern", description="helped out")],
        )
        assert ats_scorer.score_experience(jd, senior, []) > ats_scorer.score_experience(
            jd, junior, []
        )

    def test_unstated_experience_is_penalised_when_the_jd_requires_years(self):
        jd = JobDescription(job_title="Dev", min_years_experience=5.0)
        unknown = ResumeProfile(candidate_name="A", years_of_experience=None)
        assert ats_scorer.score_experience(jd, unknown, []) < 60

    def test_structure_score_rewards_a_well_formed_resume(self, strong_resume):
        assert ats_scorer.score_structure(strong_resume) >= 80

    def test_structure_score_punishes_an_unparseable_resume(self):
        bare = ResumeProfile(candidate_name="Unknown Candidate", raw_text="hello")
        assert ats_scorer.score_structure(bare) < 30

    def test_structure_score_drops_with_parse_warnings(self, strong_resume):
        clean = ats_scorer.score_structure(strong_resume, [])
        warned = ats_scorer.score_structure(strong_resume, ["partial extraction"])
        assert warned < clean

    def test_education_requires_the_right_field_not_just_a_degree(self):
        jd = JobDescription(
            job_title="ML Engineer",
            education_requirements=["Master's degree in Machine Learning"],
        )
        matching = ResumeProfile(
            candidate_name="A",
            education=[EducationEntry(degree="Master", field_of_study="Machine Learning")],
        )
        unrelated = ResumeProfile(
            candidate_name="B",
            education=[EducationEntry(degree="Master", field_of_study="History")],
        )
        assert ats_scorer.score_education(jd, matching) > ats_scorer.score_education(
            jd, unrelated
        )

    def test_no_education_at_all_scores_low(self):
        jd = JobDescription(
            job_title="Dev", education_requirements=["Bachelor's in Computer Science"]
        )
        profile = ResumeProfile(candidate_name="A")
        assert ats_scorer.score_education(jd, profile) <= 35

    def test_required_skills_component_ignores_preferred_skills(self):
        matches = skill_matcher.match_all(
            [
                JDRequirement(skill="Python", importance=SkillImportance.REQUIRED),
                JDRequirement(skill="Rust", importance=SkillImportance.PREFERRED),
            ],
            ResumeProfile(candidate_name="A", technical_skills=["Python"]),
        )
        # Python matched, Rust missing - but Rust is preferred, so 100%.
        assert ats_scorer.score_required_skills(matches) == 100.0


# --------------------------------------------------------------------------- #
# Explanation output
# --------------------------------------------------------------------------- #
class TestExplanation:
    def test_explanation_mentions_the_total(self, software_jd, strong_resume):
        ats, breakdown, mismatch, _ = score_pair(software_jd, strong_resume)
        text = ats_scorer.explain_score(breakdown, ats, mismatch)
        assert f"{ats:.0f}/100" in text

    def test_explanation_mentions_a_mismatch_when_present(
        self, mechanical_jd, strong_resume
    ):
        ats, breakdown, mismatch, _ = score_pair(mechanical_jd, strong_resume)
        text = ats_scorer.explain_score(breakdown, ats, mismatch)
        assert "mismatch" in text.lower()
