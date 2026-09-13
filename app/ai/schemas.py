"""JSON contracts for LLM responses.

The LLM is never allowed to return free prose. Each call declares the exact JSON
shape it must produce; the response is then parsed and validated into the
Pydantic models in ``app.models.models`` before anything reaches a user.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Job description extraction
# --------------------------------------------------------------------------- #
JD_SCHEMA = """{
  "job_title": "string - the role title exactly as written",
  "company": "string - company name, empty string if not stated",
  "seniority": "string - e.g. Junior, Mid, Senior, Lead. Empty if not stated",
  "required_skills": [
    {"skill": "string - a single skill/tool/technology, not a sentence",
     "context": "string - short quote from the JD showing where it is required"}
  ],
  "preferred_skills": [
    {"skill": "string - a single nice-to-have skill",
     "context": "string - short quote from the JD"}
  ],
  "min_years_experience": "number or null - minimum years explicitly required",
  "education_requirements": ["string - each stated education requirement"],
  "certifications": ["string - certifications the JD asks for"],
  "responsibilities": ["string - each key responsibility, one per item"],
  "domain_knowledge": ["string - industry/domain knowledge expected"],
  "keywords": ["string - important ATS keywords from the JD, max 30"]
}"""

# --------------------------------------------------------------------------- #
# Resume extraction
# --------------------------------------------------------------------------- #
RESUME_SCHEMA = """{
  "candidate_name": "string - the candidate's name, or 'Unknown Candidate'",
  "email": "string - empty if absent",
  "phone": "string - empty if absent",
  "location": "string - empty if absent",
  "summary": "string - the candidate's own summary/objective, empty if absent",
  "years_of_experience": "number or null - ONLY if stated or clearly derivable from dates",
  "education": [
    {"degree": "string", "field_of_study": "string",
     "institution": "string", "year": "string"}
  ],
  "certifications": ["string - each certification named in the resume"],
  "experience": [
    {"title": "string", "company": "string", "duration": "string",
     "description": "string - what they actually did"}
  ],
  "projects": [
    {"name": "string", "description": "string", "technologies": ["string"]}
  ],
  "achievements": ["string"],
  "technical_skills": ["string - every technical skill named in the resume"],
  "soft_skills": ["string"],
  "programming_languages": ["string"],
  "frameworks": ["string"],
  "databases": ["string"],
  "cloud_technologies": ["string"],
  "tools": ["string"]
}"""

# --------------------------------------------------------------------------- #
# Narrative / explanation pass
# --------------------------------------------------------------------------- #
INSIGHT_SCHEMA = """{
  "analyses": [
    {
      "candidate_name": "string - must match the candidate name given to you",
      "alignment_explanation": "string - 2-3 sentences explaining the fit overall",
      "strong_points": ["string - 2-4 specific strengths, each citing resume evidence"],
      "weak_points": ["string - 2-4 specific gaps, phrased as 'No evidence of X was found'"],
      "ats_explanation": "string - 1-2 sentences on why the ATS score landed where it did",
      "extra_missing_areas": [
        {"area": "string - a gap not captured by simple skill matching",
         "category": "technical_skill | tool | experience | domain_knowledge | certification | responsibility | project_evidence",
         "priority": "high | medium | low",
         "detail": "string - why this is a gap, referencing the JD"}
      ]
    }
  ]
}"""
