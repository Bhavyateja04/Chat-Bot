"""Gap analysis, course recommendations and the learning roadmap.

Everything here is derived from *actual* gaps found by the matcher. A course is
only ever recommended for a skill the JD asks for and the resume does not
evidence, and each recommendation carries the reason it was raised.

Course titles reference real, well-known programmes. Links are either a curated
official course page or a provider *search* URL built from the title - deep links
are never fabricated, so every URL printed actually resolves.
"""

from __future__ import annotations

from urllib.parse import quote_plus

from app.analysis import skill_matcher, skill_taxonomy as tax
from app.models.models import (
    CourseRecommendation,
    JobDescription,
    MatchStatus,
    MissingArea,
    Priority,
    ResumeProfile,
    RoadmapItem,
    SkillImportance,
    SkillMatchResult,
)

# --------------------------------------------------------------------------- #
# Curated course catalogue: canonical skill -> (title, provider, topics)
# --------------------------------------------------------------------------- #
CATALOG: dict[str, tuple[str, str, list[str]]] = {
    # cloud / devops
    "aws": ("AWS Cloud Practitioner Essentials", "AWS Skill Builder",
            ["EC2", "S3", "IAM", "VPC basics", "deploying an application"]),
    "azure": ("Microsoft Azure Fundamentals (AZ-900)", "Microsoft Learn",
              ["core Azure services", "identity", "governance", "pricing"]),
    "gcp": ("Google Cloud Digital Leader", "Google Cloud Skills Boost",
            ["Compute Engine", "Cloud Storage", "IAM", "deployment basics"]),
    "docker": ("Docker for Developers", "Docker / Udemy",
               ["images vs containers", "Dockerfile", "volumes & networking",
                "Docker Compose"]),
    "kubernetes": ("Kubernetes for the Absolute Beginners", "KodeKloud",
                   ["pods & deployments", "services", "ConfigMaps & secrets",
                    "kubectl basics"]),
    "ci/cd": ("CI/CD Fundamentals with GitHub Actions", "GitHub Learning Lab",
              ["workflow syntax", "build & test automation", "deployment stages",
               "secrets management"]),
    "terraform": ("HashiCorp Terraform Associate Preparation", "HashiCorp Learn",
                  ["HCL syntax", "providers & resources", "state management", "modules"]),
    "jenkins": ("Jenkins: Getting Started", "CloudBees / Udemy",
                ["pipelines", "Jenkinsfile", "agents", "plugin ecosystem"]),
    "linux": ("Linux Command Line Basics", "Linux Foundation",
              ["shell navigation", "permissions", "processes", "shell scripting"]),
    # languages / backend
    "python": ("Python for Everybody", "University of Michigan (Coursera)",
               ["core syntax", "data structures", "modules", "working with APIs"]),
    "java": ("Java Programming and Software Engineering Fundamentals", "Duke (Coursera)",
             ["OOP", "collections", "exceptions", "build tooling"]),
    "javascript": ("JavaScript: The Complete Guide", "Udemy",
                   ["ES6+ syntax", "async/await", "DOM", "modules"]),
    "typescript": ("Understanding TypeScript", "Udemy",
                   ["types & interfaces", "generics", "tsconfig", "migrating from JS"]),
    "golang": ("Getting Started with Go", "Google / Coursera",
               ["goroutines", "channels", "interfaces", "modules"]),
    "rest api": ("REST API Design, Development & Management", "Udemy",
                 ["resource modelling", "HTTP verbs & status codes", "versioning",
                  "authentication"]),
    "graphql": ("GraphQL Fundamentals", "Apollo Odyssey",
                ["schemas", "resolvers", "queries vs mutations", "client caching"]),
    "microservices": ("Microservices Architecture", "Coursera / Udemy",
                      ["service boundaries", "inter-service communication",
                       "resilience patterns", "observability"]),
    "django": ("Django for Everybody", "University of Michigan (Coursera)",
               ["models & ORM", "views & templates", "Django REST Framework", "auth"]),
    "flask": ("Flask Web Development", "Udemy",
              ["routing", "blueprints", "request handling", "extensions"]),
    "fastapi": ("FastAPI - The Complete Course", "Udemy",
                ["path operations", "Pydantic models", "dependency injection", "async"]),
    "node.js": ("Node.js, Express, MongoDB & More", "Udemy",
                ["event loop", "Express routing", "middleware", "npm ecosystem"]),
    "spring boot": ("Spring Boot Fundamentals", "Spring Academy",
                    ["dependency injection", "REST controllers", "Spring Data", "testing"]),
    # frontend
    "react": ("React - The Complete Guide", "Udemy",
              ["components & props", "hooks", "state management", "routing"]),
    "angular": ("Angular - The Complete Guide", "Udemy",
                ["components", "services & DI", "RxJS", "routing"]),
    "vue": ("Vue - The Complete Guide", "Udemy",
            ["reactivity", "components", "Vue Router", "Pinia"]),
    # data
    "sql": ("SQL for Data Science", "UC Davis (Coursera)",
            ["SELECT & filtering", "joins", "aggregation", "subqueries"]),
    "postgresql": ("PostgreSQL for Everybody", "University of Michigan (Coursera)",
                   ["schema design", "indexing", "query plans", "JSONB"]),
    "mysql": ("MySQL Database Administration", "Udemy",
              ["schema design", "indexing", "stored procedures", "tuning"]),
    "mongodb": ("MongoDB Basics", "MongoDB University",
                ["documents & collections", "CRUD", "aggregation pipeline", "indexing"]),
    "kafka": ("Apache Kafka Series - Learn Apache Kafka for Beginners", "Udemy",
              ["topics & partitions", "producers & consumers", "consumer groups",
               "delivery guarantees"]),
    "elasticsearch": ("Elasticsearch Engineer", "Elastic",
                      ["indexing", "query DSL", "analyzers", "aggregations"]),
    # ml / ai
    "machine learning": ("Machine Learning Specialization", "DeepLearning.AI (Coursera)",
                         ["supervised learning", "model evaluation",
                          "regularisation", "feature engineering"]),
    "deep learning": ("Deep Learning Specialization", "DeepLearning.AI (Coursera)",
                      ["neural networks", "backpropagation", "CNNs", "sequence models"]),
    "natural language processing": ("Natural Language Processing Specialization",
                                    "DeepLearning.AI (Coursera)",
                                    ["text preprocessing", "embeddings", "transformers",
                                     "sequence labelling"]),
    "computer vision": ("Introduction to Computer Vision", "Udacity",
                        ["image filtering", "feature detection", "CNNs", "segmentation"]),
    "pytorch": ("PyTorch for Deep Learning", "Udemy / freeCodeCamp",
                ["tensors", "autograd", "nn.Module", "training loops"]),
    "tensorflow": ("TensorFlow Developer Professional Certificate",
                   "DeepLearning.AI (Coursera)",
                   ["Keras API", "callbacks", "data pipelines", "model export"]),
    "scikit-learn": ("Applied Machine Learning in Python", "Michigan (Coursera)",
                     ["estimators API", "pipelines", "cross-validation", "metrics"]),
    "pandas": ("Data Analysis with Pandas", "Kaggle Learn",
               ["DataFrames", "indexing", "groupby", "joins"]),
    "numpy": ("NumPy Fundamentals", "Kaggle Learn / freeCodeCamp",
              ["ndarrays", "broadcasting", "vectorisation", "linear algebra basics"]),
    "large language models": ("Generative AI with Large Language Models",
                              "DeepLearning.AI (Coursera)",
                              ["prompting", "fine-tuning", "RAG", "evaluation"]),
    # practices / tools
    "git": ("Version Control with Git", "Atlassian (Coursera)",
            ["branching", "merging & rebasing", "pull requests", "resolving conflicts"]),
    "unit testing": ("Testing in Python with pytest", "Test Automation University",
                     ["test structure", "fixtures", "mocking", "coverage"]),
    "agile": ("Agile with Atlassian Jira", "Atlassian (Coursera)",
              ["Scrum ceremonies", "backlog grooming", "estimation", "boards"]),
    # mechanical
    "solidworks": ("SolidWorks Essentials (CSWA Preparation)", "SolidWorks / Udemy",
                   ["part modelling", "assemblies", "drawings", "sheet metal"]),
    "autocad": ("AutoCAD Essential Training", "Autodesk / LinkedIn Learning",
                ["2D drafting", "layers", "dimensioning", "plotting"]),
    "catia": ("CATIA V5 Fundamentals", "Dassault Systèmes",
              ["sketcher", "part design", "assembly design", "drafting"]),
    "creo": ("PTC Creo Parametric Fundamentals", "PTC University",
             ["sketching", "part modelling", "assemblies", "detailing"]),
    "cad": ("Computer-Aided Design Fundamentals", "Coursera",
            ["parametric modelling", "assemblies", "technical drawings"]),
    "gd&t": ("GD&T Fundamentals (ASME Y14.5)", "ASME",
             ["datums", "feature control frames", "position tolerance",
              "tolerance stack-up"]),
    "finite element analysis": ("Finite Element Analysis Fundamentals", "Coursera / NPTEL",
                                ["meshing", "boundary conditions", "solvers",
                                 "result interpretation"]),
    "ansys": ("ANSYS Workbench for Structural Analysis", "Udemy",
              ["geometry prep", "meshing", "static structural", "post-processing"]),
    "computational fluid dynamics": ("Introduction to CFD", "NPTEL / Cornell (edX)",
                                     ["governing equations", "meshing",
                                      "turbulence models", "validation"]),
    "cnc": ("CNC Machining and G-Code Programming", "Udemy",
            ["G-code basics", "tooling", "work offsets", "machining strategies"]),
    "six sigma": ("Six Sigma Green Belt Certification", "ASQ / Coursera",
                  ["DMAIC", "process capability", "control charts", "root cause analysis"]),
    "matlab": ("MATLAB Onramp", "MathWorks",
               ["matrices", "scripts", "plotting", "Simulink basics"]),
    # electrical
    "plc": ("PLC Programming from Scratch", "Udemy",
            ["ladder logic", "I/O addressing", "timers & counters", "HMI basics"]),
    "verilog": ("Digital Design with Verilog", "NPTEL / Udemy",
                ["combinational logic", "sequential logic", "testbenches", "synthesis"]),
    "embedded systems": ("Embedded Systems Programming in C", "Udemy",
                         ["microcontroller architecture", "GPIO", "interrupts", "timers"]),
    # business / analytics
    "power bi": ("Microsoft Power BI Data Analyst (PL-300)", "Microsoft Learn",
                 ["data modelling", "DAX", "visualisations", "publishing reports"]),
    "tableau": ("Tableau Desktop Specialist", "Tableau / Coursera",
                ["connecting data", "calculated fields", "dashboards", "publishing"]),
    "excel": ("Excel Skills for Business", "Macquarie University (Coursera)",
              ["formulas", "pivot tables", "lookup functions", "charts"]),
    "seo": ("SEO Specialization", "UC Davis (Coursera)",
            ["keyword research", "on-page SEO", "link building", "analytics"]),
}

# --------------------------------------------------------------------------- #
# Course links
# --------------------------------------------------------------------------- #
# Two rules keep these honest:
#   1. COURSE_URLS holds only stable, official course or curriculum pages.
#   2. Anything not listed falls back to a provider *search* URL built from the
#      course title. A search link always resolves, so the bot never emits a
#      fabricated deep link that 404s in front of a recruiter.
COURSE_URLS: dict[str, str] = {
    # cloud / devops
    "aws": "https://skillbuilder.aws/",
    "azure": "https://learn.microsoft.com/en-us/training/courses/az-900t00",
    "gcp": "https://www.cloudskillsboost.google/",
    "docker": "https://docs.docker.com/get-started/",
    "kubernetes": "https://kodekloud.com/courses/kubernetes-for-the-absolute-beginners-hands-on/",
    "ci/cd": "https://docs.github.com/en/actions",
    "terraform": "https://developer.hashicorp.com/terraform/tutorials",
    "jenkins": "https://www.jenkins.io/doc/tutorials/",
    "linux": "https://training.linuxfoundation.org/",
    # languages / backend
    "python": "https://www.coursera.org/specializations/python",
    "java": "https://www.coursera.org/specializations/java-programming",
    "golang": "https://go.dev/learn/",
    "graphql": "https://www.apollographql.com/tutorials/",
    "django": "https://www.coursera.org/specializations/django",
    "flask": "https://flask.palletsprojects.com/en/stable/tutorial/",
    "fastapi": "https://fastapi.tiangolo.com/tutorial/",
    "spring boot": "https://spring.academy/",
    # data
    "sql": "https://www.coursera.org/learn/sql-for-data-science",
    "postgresql": "https://www.postgresql.org/docs/current/tutorial.html",
    "mongodb": "https://learn.mongodb.com/",
    "elasticsearch": "https://www.elastic.co/training/",
    # ml / ai
    "machine learning": "https://www.coursera.org/specializations/machine-learning-introduction",
    "deep learning": "https://www.coursera.org/specializations/deep-learning",
    "natural language processing": "https://www.coursera.org/specializations/natural-language-processing",
    "pytorch": "https://pytorch.org/tutorials/",
    "tensorflow": "https://www.tensorflow.org/tutorials",
    "scikit-learn": "https://scikit-learn.org/stable/tutorial/index.html",
    "pandas": "https://www.kaggle.com/learn/pandas",
    "numpy": "https://numpy.org/learn/",
    "large language models": "https://www.deeplearning.ai/courses/",
    # practices / tools
    "git": "https://www.coursera.org/learn/version-control-with-git",
    "unit testing": "https://docs.pytest.org/en/stable/getting-started.html",
    "agile": "https://www.coursera.org/specializations/atlassian-agile-development",
    # mechanical
    "solidworks": "https://www.solidworks.com/solidworks-certification-program",
    "autocad": "https://www.autodesk.com/learning",
    "catia": "https://www.3ds.com/learn/training",
    "creo": "https://www.ptc.com/en/education",
    "gd&t": "https://www.asme.org/codes-standards/find-codes-standards/y14-5-dimensioning-tolerancing",
    "ansys": "https://www.ansys.com/academic/learning-resources",
    "finite element analysis": "https://nptel.ac.in/",
    "computational fluid dynamics": "https://nptel.ac.in/",
    "six sigma": "https://asq.org/cert/six-sigma-green-belt",
    "matlab": "https://matlabacademy.mathworks.com/",
    # electrical
    "verilog": "https://nptel.ac.in/",
    # business / analytics
    "power bi": "https://learn.microsoft.com/en-us/credentials/certifications/data-analyst-associate/",
    "tableau": "https://www.tableau.com/learn/training",
    "excel": "https://www.coursera.org/specializations/excel",
    "seo": "https://www.coursera.org/specializations/seo",
}

# Provider -> search URL template. Used when there is no curated link.
_PROVIDER_SEARCH: dict[str, str] = {
    "udemy": "https://www.udemy.com/courses/search/?q={query}",
    "coursera": "https://www.coursera.org/search?query={query}",
    "kaggle": "https://www.kaggle.com/learn",
    "edx": "https://www.edx.org/search?q={query}",
    "udacity": "https://www.udacity.com/catalog?searchValue={query}",
    "linkedin": "https://www.linkedin.com/learning/search?keywords={query}",
    "microsoft": "https://learn.microsoft.com/en-us/search/?terms={query}",
    "nptel": "https://nptel.ac.in/",
    "freecodecamp": "https://www.freecodecamp.org/learn",
}

# Last resort: a Coursera search, which covers most professional topics.
_DEFAULT_SEARCH = "https://www.coursera.org/search?query={query}"


def course_url(skill: str, title: str, provider: str) -> str:
    """A link for a recommended course.

    Returns a curated official page when one is known for the skill, otherwise a
    provider search URL built from the course title. Deep links are never
    fabricated, so every URL the bot prints actually resolves.
    """
    canon = tax.canonical(skill)
    curated = COURSE_URLS.get(canon)
    if curated:
        return curated

    query = quote_plus(title.split(" - ")[0].split(" (")[0].strip() or skill)
    provider_key = provider.lower()
    for name, template in _PROVIDER_SEARCH.items():
        if name in provider_key:
            return template.format(query=query)
    return _DEFAULT_SEARCH.format(query=query)


# Generic topics used when a skill is not in the catalogue.
_GENERIC_TOPICS = ["core concepts", "hands-on practice", "a small portfolio project"]

_MAX_COURSES = 6
_MAX_ROADMAP = 4


# --------------------------------------------------------------------------- #
# Missing areas
# --------------------------------------------------------------------------- #
def build_missing_areas(
    jd: JobDescription,
    profile: ResumeProfile,
    matches: list[SkillMatchResult],
    alignment=None,
) -> list[MissingArea]:
    """Turn gaps into prioritised, categorised missing areas.

    Goes beyond missing keywords: experience shortfall, education, certifications,
    domain knowledge and un-evidenced responsibilities are all considered.
    """
    areas: list[MissingArea] = []
    jd_domain = jd.detected_domain or tax.classify_domain(jd.raw_text)[0]
    core_vocabulary = tax.canonical_set(tax.DOMAIN_VOCABULARY.get(jd_domain, set()))

    # --- 1. missing skills / tools ---
    for match in matches:
        if match.status != MatchStatus.MISSING:
            continue
        canon = tax.canonical(match.skill)
        if match.importance == SkillImportance.REQUIRED:
            priority = Priority.HIGH if canon in core_vocabulary else Priority.MEDIUM
            detail = (
                f"'{tax.display_name(match.skill)}' is listed as a required skill in "
                f"the job description, but no evidence of it was found in the resume."
            )
        else:
            priority = Priority.LOW
            detail = (
                f"'{tax.display_name(match.skill)}' is a preferred (nice-to-have) "
                f"skill that the resume does not demonstrate."
            )
        areas.append(
            MissingArea(
                area=match.skill,
                category="technical_skill",
                priority=priority,
                detail=detail,
            )
        )

    # --- 2. partial matches worth strengthening ---
    for match in matches:
        if match.status != MatchStatus.PARTIAL:
            continue
        if match.importance != SkillImportance.REQUIRED:
            continue
        areas.append(
            MissingArea(
                area=f"{tax.display_name(match.skill)} (only related experience shown)",
                category="technical_skill",
                priority=Priority.MEDIUM,
                detail=(
                    f"The JD requires '{tax.display_name(match.skill)}'. The resume "
                    f"shows '{tax.display_name(match.matched_via)}' ({match.relation}), "
                    f"which is related but not the same technology."
                ),
            )
        )

    # --- 3. experience shortfall ---
    if jd.min_years_experience is not None:
        if profile.years_of_experience is None:
            areas.append(
                MissingArea(
                    area=f"Evidence of {jd.min_years_experience:.0f}+ years of experience",
                    category="experience",
                    priority=Priority.MEDIUM,
                    detail=(
                        f"The JD asks for at least {jd.min_years_experience:.0f} years. "
                        "The resume does not state a total years-of-experience figure, "
                        "so this could not be verified."
                    ),
                )
            )
        elif profile.years_of_experience < jd.min_years_experience:
            gap = jd.min_years_experience - profile.years_of_experience
            areas.append(
                MissingArea(
                    area=(
                        f"{gap:.0f} more year(s) of experience "
                        f"(JD asks for {jd.min_years_experience:.0f}+)"
                    ),
                    category="experience",
                    priority=Priority.HIGH if gap >= 2 else Priority.MEDIUM,
                    detail=(
                        f"The resume evidences about {profile.years_of_experience:.0f} "
                        f"years against a stated requirement of "
                        f"{jd.min_years_experience:.0f}+ years."
                    ),
                )
            )

    # --- 4. certifications ---
    resume_certs = tax.canonical_set(profile.certifications)
    resume_text_norm = tax.normalize(profile.raw_text)
    for certification in jd.certifications:
        if (
            tax.canonical(certification) not in resume_certs
            and tax.normalize(certification) not in resume_text_norm
        ):
            areas.append(
                MissingArea(
                    area=certification,
                    category="certification",
                    priority=Priority.MEDIUM,
                    detail=f"The JD mentions the '{certification}' certification, "
                           "which does not appear in the resume.",
                )
            )

    # --- 5. education ---
    if jd.education_requirements and not profile.education:
        areas.append(
            MissingArea(
                area="Education details",
                category="education",
                priority=Priority.MEDIUM,
                detail=(
                    "The JD states an education requirement "
                    f"('{jd.education_requirements[0][:80]}'), but no education "
                    "section was found in the resume."
                ),
            )
        )

    # --- 6. domain knowledge ---
    for domain_item in jd.domain_knowledge:
        if tax.normalize(domain_item) not in resume_text_norm:
            areas.append(
                MissingArea(
                    area=domain_item,
                    category="domain_knowledge",
                    priority=Priority.MEDIUM,
                    detail=f"The JD expects domain knowledge of '{domain_item}', "
                           "which is not evidenced in the resume.",
                )
            )

    # --- 7. responsibilities with no supporting evidence ---
    uncovered = _uncovered_responsibilities(jd, profile)
    for responsibility in uncovered[:3]:
        areas.append(
            MissingArea(
                area=_shorten(responsibility),
                category="responsibility",
                priority=Priority.MEDIUM,
                detail=(
                    "This JD responsibility has no clear supporting evidence in the "
                    "resume: " + _shorten(responsibility, 160)
                ),
            )
        )

    # --- 8. no project evidence at all ---
    if not profile.projects and not profile.experience:
        areas.append(
            MissingArea(
                area="Demonstrated project or work evidence",
                category="project_evidence",
                priority=Priority.HIGH,
                detail="No work experience or project section could be identified, "
                       "so none of the claimed skills are backed by evidence.",
            )
        )

    # --- 9. alignment-dimension gaps ---
    # A resume can match every named skill and still align poorly on experience,
    # project evidence, keywords or education. Those dimensions are gaps too.
    areas.extend(_alignment_gaps(jd, profile, matches, alignment, areas))

    order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    areas.sort(key=lambda a: order[a.priority])
    return areas


# --------------------------------------------------------------------------- #
# Alignment-dimension gaps
# --------------------------------------------------------------------------- #
# A dimension at or above this is considered aligned and raises no gap.
ALIGNMENT_OK = 75.0


def _alignment_priority(score: float) -> Priority:
    if score < 40.0:
        return Priority.HIGH
    if score < 60.0:
        return Priority.MEDIUM
    return Priority.LOW


def _alignment_gaps(
    jd: JobDescription,
    profile: ResumeProfile,
    matches: list[SkillMatchResult],
    alignment,
    existing: list[MissingArea],
) -> list[MissingArea]:
    """Turn weak alignment dimensions into evidence-backed gaps.

    Each gap must be justified by something concrete in the two documents - a
    stated requirement, an absent keyword, a missing section. A dimension that
    scores low for a reason we cannot point at is left out rather than invented.
    """
    if alignment is None:
        return []

    gaps: list[MissingArea] = []
    categories_present = {a.category for a in existing}

    # --- experience relevance ---
    if alignment.experience < ALIGNMENT_OK:
        reasons: list[str] = []
        # The years shortfall may already be reported; don't say it twice.
        if "experience" not in categories_present:
            if jd.min_years_experience is not None and profile.years_of_experience is None:
                reasons.append(
                    f"the JD asks for {jd.min_years_experience:.0f}+ years but the "
                    "resume does not state a total"
                )
            elif (
                jd.min_years_experience is not None
                and profile.years_of_experience is not None
                and profile.years_of_experience < jd.min_years_experience
            ):
                reasons.append(
                    f"the resume evidences about {profile.years_of_experience:.0f} "
                    f"years against {jd.min_years_experience:.0f}+ required"
                )
        if not profile.experience:
            reasons.append("no work-experience section could be identified")
        else:
            evidenced = _skills_evidenced_in(
                " ".join(
                    f"{e.title} {e.company} {e.description}" for e in profile.experience
                ),
                matches,
            )
            if matches and evidenced < len(matches):
                reasons.append(
                    f"only {evidenced} of {len(matches)} JD skills appear inside the "
                    "work-experience descriptions"
                )
        if reasons:
            gaps.append(
                MissingArea(
                    area=f"Experience alignment ({alignment.experience:.0f}%)",
                    category="experience",
                    priority=_alignment_priority(alignment.experience),
                    detail=(
                        "The JD's experience expectations are only partly evidenced: "
                        + "; ".join(reasons)
                        + "."
                    ),
                )
            )

    # --- project / applied evidence ---
    if alignment.projects < ALIGNMENT_OK:
        reasons = []
        if not profile.projects:
            reasons.append("no projects section could be identified")
        applied_text = " ".join(
            f"{p.name} {p.description} {' '.join(p.technologies)}"
            for p in profile.projects
        ) + " " + " ".join(e.description for e in profile.experience)
        evidenced = _skills_evidenced_in(applied_text, matches)
        if matches and evidenced < len(matches):
            reasons.append(
                f"only {evidenced} of {len(matches)} JD skills are demonstrated in "
                "project or work descriptions"
            )
        if jd.responsibilities:
            uncovered = _uncovered_responsibilities(jd, profile)
            if uncovered:
                reasons.append(
                    f"{len(uncovered)} of {len(jd.responsibilities)} JD "
                    "responsibilities have no supporting project evidence"
                )
        if reasons:
            gaps.append(
                MissingArea(
                    area=f"Project relevance ({alignment.projects:.0f}%)",
                    category="project_evidence",
                    priority=_alignment_priority(alignment.projects),
                    detail=(
                        "Projects and work history do not yet evidence what the JD "
                        "describes: " + "; ".join(reasons) + "."
                    ),
                )
            )

    # --- responsibilities ---
    if alignment.responsibilities < ALIGNMENT_OK and jd.responsibilities:
        uncovered = _uncovered_responsibilities(jd, profile)
        if uncovered:
            gaps.append(
                MissingArea(
                    area=f"Responsibility coverage ({alignment.responsibilities:.0f}%)",
                    category="responsibility",
                    priority=_alignment_priority(alignment.responsibilities),
                    detail=(
                        f"{len(uncovered)} of {len(jd.responsibilities)} "
                        "responsibilities listed in the JD are not clearly reflected "
                        f"anywhere in the resume, starting with: "
                        f"{_shorten(uncovered[0], 90)}"
                    ),
                )
            )

    # --- keyword coverage ---
    if alignment.keywords < ALIGNMENT_OK:
        absent = skill_matcher.missing_keywords(
            jd.keywords or [r.skill for r in jd.all_requirements()], profile
        )
        if absent:
            shown = ", ".join(tax.display_name(k) for k in absent[:8])
            gaps.append(
                MissingArea(
                    area=f"Keyword coverage ({alignment.keywords:.0f}%)",
                    category="keywords",
                    priority=_alignment_priority(alignment.keywords),
                    detail=(
                        f"{len(absent)} keyword(s) used in the JD do not appear in the "
                        f"resume, including: {shown}. ATS filters commonly screen on "
                        "this exact vocabulary."
                    ),
                )
            )

    # --- education ---
    if alignment.education < ALIGNMENT_OK and "education" not in categories_present:
        if jd.education_requirements or jd.certifications:
            if profile.education:
                degrees = "; ".join(
                    " ".join(filter(None, [e.degree, e.field_of_study])).strip()
                    for e in profile.education[:2]
                )
                detail = (
                    f"The JD states: \"{_shorten(jd.education_requirements[0], 90)}\". "
                    f"The resume shows: {degrees or 'an unspecified degree'}, which "
                    "only partly matches."
                ) if jd.education_requirements else (
                    "The certifications the JD names are not all evidenced in the resume."
                )
            else:
                detail = (
                    "The JD states an education requirement, but no education section "
                    "could be identified in the resume."
                )
            gaps.append(
                MissingArea(
                    area=f"Education match ({alignment.education:.0f}%)",
                    category="education",
                    priority=_alignment_priority(alignment.education),
                    detail=detail,
                )
            )

    return gaps


def _skills_evidenced_in(text: str, matches: list[SkillMatchResult]) -> int:
    """How many matched JD skills actually appear inside ``text``."""
    if not text.strip():
        return 0
    norm = " " + tax.normalize(text) + " "
    count = 0
    for match in matches:
        if match.status == MatchStatus.MISSING:
            continue
        needle = tax.normalize(match.matched_via or match.skill)
        if needle and needle in norm:
            count += 1
    return count


def _uncovered_responsibilities(jd: JobDescription, profile: ResumeProfile) -> list[str]:
    """JD responsibilities with no meaningful word overlap in the resume."""
    if not jd.responsibilities:
        return []
    resume_text = tax.normalize(profile.raw_text)
    stopwords = {
        "with", "and", "the", "for", "our", "you", "will", "work", "team", "using",
        "across", "into", "from", "that", "this", "their", "them", "other", "also",
        "have", "been", "your", "within", "ensure", "help", "through",
    }
    uncovered: list[str] = []
    for responsibility in jd.responsibilities:
        tokens = [
            t for t in tax.normalize(responsibility).split()
            if len(t) > 3 and t not in stopwords
        ]
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in resume_text)
        if hits < max(1, len(tokens) // 4):
            uncovered.append(responsibility)
    return uncovered


def _shorten(text: str, limit: int = 70) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------- #
# Courses
# --------------------------------------------------------------------------- #
def recommend_courses(
    matches: list[SkillMatchResult],
    missing_areas: list[MissingArea],
    keyword_gaps: list[tuple[str, Priority, str]] | None = None,
) -> list[CourseRecommendation]:
    """Recommend courses strictly for demonstrated skill gaps, highest priority first."""
    gaps: list[tuple[str, Priority, str]] = []
    seen: set[str] = set()

    # Required missing skills come first.
    for match in matches:
        if match.status != MatchStatus.MISSING:
            continue
        canon = tax.canonical(match.skill)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        if match.importance == SkillImportance.REQUIRED:
            reason = (
                f"{tax.display_name(match.skill)} is listed as a required skill in the "
                f"JD, but no evidence of it was found in the resume."
            )
            gaps.append((match.skill, Priority.HIGH, reason))
        else:
            reason = (
                f"{tax.display_name(match.skill)} is a preferred skill in the JD that "
                f"the resume does not demonstrate."
            )
            gaps.append((match.skill, Priority.LOW, reason))

    # Required skills only partially covered are worth a targeted course too.
    for match in matches:
        if match.status != MatchStatus.PARTIAL:
            continue
        if match.importance != SkillImportance.REQUIRED:
            continue
        canon = tax.canonical(match.skill)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        gaps.append(
            (
                match.skill,
                Priority.MEDIUM,
                f"The JD requires {tax.display_name(match.skill)}; the resume only "
                f"shows the related '{tax.display_name(match.matched_via)}' "
                f"({match.relation}).",
            )
        )

    # Certifications and domain knowledge flagged as missing.
    for area in missing_areas:
        if area.category not in {"certification", "domain_knowledge"}:
            continue
        canon = tax.canonical(area.area)
        if canon and canon not in seen:
            seen.add(canon)
            gaps.append((area.area, area.priority, area.detail))

    # Knowledge gaps implied by weak keyword coverage: a JD keyword that is a
    # recognised skill and is absent from the resume is a genuine learning gap,
    # even when every skill the JD listed explicitly was matched.
    for keyword, priority, reason in keyword_gaps or []:
        canon = tax.canonical(keyword)
        if canon and canon not in seen:
            seen.add(canon)
            gaps.append((keyword, priority, reason))

    order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    gaps.sort(key=lambda g: order[g[1]])

    recommendations: list[CourseRecommendation] = []
    for skill, priority, reason in gaps[:_MAX_COURSES]:
        title, provider, _ = _catalog_entry(skill)
        recommendations.append(
            CourseRecommendation(
                title=title,
                provider=provider,
                skill=skill,
                reason=reason,
                priority=priority,
                url=course_url(skill, title, provider),
            )
        )
    return recommendations


def keyword_learning_gaps(
    jd: JobDescription, profile: ResumeProfile, alignment=None
) -> list[tuple[str, Priority, str]]:
    """Absent JD keywords that are recognised skills, as course-worthy gaps.

    Only keywords the taxonomy recognises as real skills qualify - ordinary JD
    prose ("collaborate", "stakeholder") is not something to take a course in.
    Returns ``(skill, priority, reason)`` triples.
    """
    if alignment is not None and alignment.keywords >= ALIGNMENT_OK:
        return []

    absent = skill_matcher.missing_keywords(jd.keywords, profile)
    priority = (
        _alignment_priority(alignment.keywords) if alignment is not None else Priority.LOW
    )

    out: list[tuple[str, Priority, str]] = []
    for keyword in absent:
        canon = tax.canonical(keyword)
        # Only genuine, catalogued technologies - not generic JD vocabulary.
        if canon not in tax.ALIASES and canon not in CATALOG:
            continue
        out.append(
            (
                keyword,
                priority,
                f"{tax.display_name(keyword)} appears in the job description but no "
                "evidence of it was found in the resume.",
            )
        )
    return out[:4]


def _catalog_entry(skill: str) -> tuple[str, str, list[str]]:
    """Look up a curated course, or synthesise a sensible generic one."""
    canon = tax.canonical(skill)
    if canon in CATALOG:
        return CATALOG[canon]
    pretty = skill.strip().title() if skill.islower() else skill.strip()
    return (f"{pretty} - Fundamentals", "", list(_GENERIC_TOPICS))


# --------------------------------------------------------------------------- #
# Learning roadmap
# --------------------------------------------------------------------------- #
def build_roadmap(
    matches: list[SkillMatchResult],
    courses: list[CourseRecommendation],
) -> list[RoadmapItem]:
    """A short, prioritised 'what to actually learn' plan."""
    items: list[RoadmapItem] = []
    seen: set[str] = set()

    ordered = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    for course in sorted(courses, key=lambda c: ordered[c.priority]):
        canon = tax.canonical(course.skill)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        _, _, topics = _catalog_entry(course.skill)
        items.append(
            RoadmapItem(
                priority=len(items) + 1,
                skill=course.skill,
                topics=topics[:4],
                why=course.reason,
            )
        )
        if len(items) >= _MAX_ROADMAP:
            break
    return items


# --------------------------------------------------------------------------- #
# Resume improvement suggestions (secondary feature)
# --------------------------------------------------------------------------- #
def improvement_suggestions(
    jd: JobDescription,
    profile: ResumeProfile,
    matches: list[SkillMatchResult],
) -> list[str]:
    """Top practical fixes for the resume itself, not the candidate's skills."""
    suggestions: list[str] = []

    if not profile.email:
        suggestions.append(
            "Add an email address near the top - many ATS parsers reject resumes "
            "with no machine-readable contact details."
        )
    if not profile.all_skills():
        suggestions.append(
            "Add an explicit 'Technical Skills' section - keyword based screening "
            "relies on it heavily."
        )
    if not profile.projects and not profile.experience:
        suggestions.append(
            "Add a work experience or projects section so that the listed skills "
            "are backed by evidence."
        )

    exact_required = [
        m for m in matches
        if m.status == MatchStatus.EXACT and m.importance == SkillImportance.REQUIRED
    ]
    unevidenced = [
        m.skill for m in exact_required
        if not m.evidence or len(m.evidence) < 25
    ]
    if unevidenced:
        suggestions.append(
            "Show these required skills inside a bullet describing real work, not "
            "just in the skills list: "
            + ", ".join(tax.display_name(s) for s in unevidenced[:4]) + "."
        )

    if profile.years_of_experience is None:
        suggestions.append(
            "State total years of experience in the summary line - recruiters and "
            "ATS filters both screen on it."
        )

    missing_required = [
        m.skill for m in matches
        if m.status == MatchStatus.MISSING and m.importance == SkillImportance.REQUIRED
    ]
    if missing_required:
        suggestions.append(
            "Mirror the JD's exact wording for any of these you genuinely have "
            "experience with: "
            + ", ".join(tax.display_name(s) for s in missing_required[:4]) + "."
        )

    return suggestions[:3]
