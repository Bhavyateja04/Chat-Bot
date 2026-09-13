"""Structured extraction from resumes and job descriptions (LLM-free path)."""

from __future__ import annotations

import pytest

from app.analysis.heuristic_extractor import extract_jd, extract_resume, find_skills
from app.parsers.text_parser import from_string


# --------------------------------------------------------------------------- #
# Resume extraction
# --------------------------------------------------------------------------- #
class TestResumeExtraction:
    def test_candidate_name(self, strong_resume):
        assert strong_resume.candidate_name == "Priya Raghavan"

    def test_contact_details(self, strong_resume):
        assert strong_resume.email == "priya.raghavan@example.com"
        assert "98765" in strong_resume.phone

    def test_years_of_experience(self, strong_resume):
        assert strong_resume.years_of_experience == 7.0

    def test_education(self, strong_resume):
        assert strong_resume.education
        degrees = " ".join(e.degree for e in strong_resume.education).lower()
        assert "bachelor" in degrees

    def test_experience_entries(self, strong_resume):
        assert len(strong_resume.experience) >= 2
        titles = " ".join(e.title for e in strong_resume.experience).lower()
        assert "engineer" in titles
        assert all(e.description for e in strong_resume.experience)

    def test_projects_are_not_fragmented_by_line_wrapping(self, strong_resume):
        """Wrapped bullet text must not be mistaken for a new project title."""
        assert len(strong_resume.projects) == 2
        names = [p.name for p in strong_resume.projects]
        assert "Distributed Rate Limiter" in names
        assert "Observability Dashboard" in names

    def test_project_technologies_are_detected(self, strong_resume):
        technologies = {
            t.lower() for p in strong_resume.projects for t in p.technologies
        }
        assert "redis" in technologies or "python" in technologies

    def test_certifications(self, strong_resume):
        blob = " ".join(strong_resume.certifications).lower()
        assert "aws" in blob

    def test_skills_are_bucketed(self, strong_resume):
        assert "python" in [s.lower() for s in strong_resume.programming_languages]
        assert "postgresql" in [s.lower() for s in strong_resume.databases]
        assert "aws" in [s.lower() for s in strong_resume.cloud_technologies]

    def test_soft_skills_are_detected_only_when_actually_present(self):
        present = extract_resume(
            from_string(
                "Dana Roy\ndana@example.com\nStrong communication and leadership, "
                "with a track record of mentoring engineers.",
                "dana.txt",
            )
        )
        assert {"communication", "leadership", "mentoring"} <= set(present.soft_skills)

    def test_soft_skills_are_not_invented(self, strong_resume):
        """This resume never uses soft-skill vocabulary, so none may be claimed."""
        assert strong_resume.soft_skills == []

    def test_domain_is_detected(self, strong_resume, mechanical_resume):
        assert strong_resume.detected_domain == "software_engineering"
        assert mechanical_resume.detected_domain == "mechanical_engineering"

    def test_mechanical_resume_extraction(self, mechanical_resume):
        assert mechanical_resume.candidate_name == "Rohit Deshmukh"
        assert mechanical_resume.years_of_experience == 6.0
        skills = [s.lower() for s in mechanical_resume.all_skills()]
        assert "solidworks" in skills
        assert "gd&t" in skills

    def test_no_skills_are_invented(self):
        """Nothing may be extracted that is not in the document."""
        document = from_string(
            "Jane Doe\njane@example.com\n\nExperience\nSoftware Engineer, 2020 - 2023\n"
            "- Wrote Python scripts for data processing\n",
            "jane.txt",
        )
        profile = extract_resume(document)
        skills = {s.lower() for s in profile.all_skills()}

        assert "python" in skills
        for absent in ("docker", "kubernetes", "aws", "react"):
            assert absent not in skills, f"Invented skill: {absent}"

    def test_an_empty_document_yields_an_empty_profile(self):
        profile = extract_resume(from_string("", "empty.txt"))
        assert profile.candidate_name == "Unknown Candidate"
        assert profile.all_skills() == []
        assert not profile.has_structure()

    def test_years_are_inferred_from_date_ranges(self):
        document = from_string(
            "Sam Patel\nsam@example.com\n\nWork Experience\n"
            "Engineer, Acme, 2018 - 2021\n- Built services\n",
            "sam.txt",
        )
        profile = extract_resume(document)
        assert profile.years_of_experience == 3.0

    def test_a_year_is_not_mistaken_for_a_phone_number(self):
        document = from_string(
            "Alex Kim\nalex@example.com\n\nEducation\nBachelor of Science, 2019\n",
            "alex.txt",
        )
        profile = extract_resume(document)
        assert profile.phone == ""


# --------------------------------------------------------------------------- #
# Skill detection
# --------------------------------------------------------------------------- #
class TestSkillDetection:
    def test_finds_skills_present_in_text(self):
        skills = find_skills("Experienced with Python, Docker and PostgreSQL.")
        assert {"python", "docker", "postgresql"} <= set(skills)

    def test_does_not_find_absent_skills(self):
        assert "kubernetes" not in find_skills("I write Python and SQL.")

    def test_synonyms_are_normalised_on_detection(self):
        skills = find_skills("Worked with JS, Postgres and K8s.")
        assert {"javascript", "postgresql", "kubernetes"} <= set(skills)

    def test_empty_text_finds_nothing(self):
        assert find_skills("") == []

    def test_short_tokens_do_not_match_inside_words(self):
        assert "golang" not in find_skills("We had a good goal at Google.")


# --------------------------------------------------------------------------- #
# Job description extraction
# --------------------------------------------------------------------------- #
class TestJdExtraction:
    def test_job_title(self, software_jd):
        assert "Software Engineer" in software_jd.job_title

    def test_minimum_years(self, software_jd):
        assert software_jd.min_years_experience == 5.0

    def test_required_and_preferred_are_separated(self, software_jd):
        required = {r.skill.lower() for r in software_jd.required_skills}
        preferred = {r.skill.lower() for r in software_jd.preferred_skills}

        assert "python" in required
        assert "docker" in required
        assert "kubernetes" in preferred
        assert "terraform" in preferred
        assert not (required & preferred), "A skill cannot be both"

    def test_responsibilities(self, software_jd):
        assert len(software_jd.responsibilities) >= 5

    def test_education_requirements(self, software_jd):
        assert software_jd.education_requirements
        assert "bachelor" in software_jd.education_requirements[0].lower()

    def test_keywords_are_produced(self, software_jd):
        assert len(software_jd.keywords) >= 10

    def test_jd_is_considered_complete(self, software_jd):
        assert software_jd.is_complete()

    def test_mechanical_jd_extraction(self, mechanical_jd):
        assert "Mechanical" in mechanical_jd.job_title
        required = {r.skill.lower() for r in mechanical_jd.required_skills}
        assert "solidworks" in required
        assert "gd&t" in required
        assert mechanical_jd.detected_domain == "mechanical_engineering"

    def test_ml_jd_extraction(self, ml_jd):
        required = {r.skill.lower() for r in ml_jd.required_skills}
        assert "python" in required
        assert any("learning" in s or "pytorch" in s for s in required)
        assert ml_jd.detected_domain in {"data_science_ml", "software_engineering"}

    def test_a_vague_jd_is_not_marked_complete(self):
        jd = extract_jd(from_string("We want someone great.", "vague.txt"))
        assert not jd.is_complete() or not jd.required_skills

    def test_an_empty_jd_does_not_crash(self):
        jd = extract_jd(from_string("", "empty.txt"))
        assert jd.job_title
        assert jd.all_requirements() == []
