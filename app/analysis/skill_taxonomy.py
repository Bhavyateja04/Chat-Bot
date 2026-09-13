"""Skill knowledge base: synonyms, related-skill graph and domain vocabularies.

This is deliberately deterministic and hand-curated rather than LLM-generated so
that scoring is reproducible and explainable. Three structures live here:

1. ``ALIASES``  - alias -> canonical skill.  "js" and "javascript" are the SAME
   skill, so a resume saying "JS" fully satisfies a JD asking for "JavaScript".

2. ``RELATED``  - canonical -> {canonical: relation}.  These are *transferable but
   not equivalent* skills.  Django is related to Flask (both Python web
   frameworks) so it earns partial credit, never full credit.  The graph is
   intentionally conservative: Python is NOT related to Java, React is NOT
   related to Angular for credit purposes beyond "same category".

3. ``DOMAIN_VOCABULARY`` - domain -> weighted terms, used to classify a JD or a
   resume into an engineering/professional domain so that a mechanical JD paired
   with a software resume is caught as a role mismatch.
"""

from __future__ import annotations

import re
from typing import Iterable

# --------------------------------------------------------------------------- #
# 1. Synonyms / aliases -> canonical form
# --------------------------------------------------------------------------- #
_ALIAS_GROUPS: dict[str, list[str]] = {
    # languages
    "javascript": ["js", "java script", "ecmascript", "es6", "es2015"],
    "typescript": ["ts"],
    "python": ["python3", "python 3", "py"],
    "c++": ["cpp", "cplusplus", "c plus plus"],
    "c#": ["csharp", "c sharp", ".net c#"],
    "golang": ["go lang", "go"],
    "ruby": ["ruby lang"],
    # web / frameworks
    "react": ["react.js", "reactjs", "react js"],
    "angular": ["angular.js", "angularjs", "angular js", "angular 2+"],
    "vue": ["vue.js", "vuejs", "vue js"],
    "node.js": ["node", "nodejs", "node js"],
    "express": ["express.js", "expressjs"],
    "next.js": ["nextjs", "next js"],
    "spring boot": ["springboot", "spring-boot"],
    "asp.net": ["aspnet", "asp .net"],
    "rest api": [
        "rest apis",
        "restful api",
        "restful apis",
        "restful services",
        "rest",
        "restful",
        "rest services",
    ],
    "graphql": ["graph ql"],
    "microservices": ["micro services", "microservice architecture", "micro-services"],
    # data / db
    "postgresql": ["postgres", "postgre sql", "psql"],
    "mysql": ["my sql"],
    "sql server": ["mssql", "ms sql", "microsoft sql server", "t-sql", "tsql"],
    "mongodb": ["mongo", "mongo db"],
    "elasticsearch": ["elastic search", "elk", "opensearch"],
    "sql": ["structured query language"],
    "nosql": ["no sql", "non-relational database"],
    # cloud / devops
    "aws": ["amazon web services", "amazon aws"],
    "azure": ["microsoft azure", "ms azure", "azure cloud"],
    "gcp": ["google cloud", "google cloud platform"],
    "kubernetes": ["k8s", "kube"],
    "docker": ["containerization", "containerisation", "containers"],
    "ci/cd": [
        "cicd",
        "ci cd",
        "continuous integration",
        "continuous delivery",
        "continuous deployment",
        "ci/cd pipelines",
        "build pipelines",
    ],
    "terraform": ["hashicorp terraform", "infrastructure as code", "iac"],
    "github actions": ["gh actions", "github workflow"],
    "jenkins": ["jenkins ci"],
    # ai / ml
    "machine learning": ["ml", "machine-learning"],
    "deep learning": ["dl", "neural networks", "neural network"],
    "natural language processing": ["nlp", "natural-language processing"],
    "computer vision": ["cv", "image processing", "opencv"],
    "tensorflow": ["tensor flow", "tf", "keras"],
    "pytorch": ["torch", "py torch"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "large language models": ["llm", "llms", "genai", "generative ai", "gpt"],
    "pandas": ["pandas library"],
    "numpy": ["num py"],
    # tools / practice
    "git": ["github", "gitlab", "bitbucket", "version control"],
    "agile": ["scrum", "agile methodology", "agile methodologies", "kanban"],
    "jira": ["atlassian jira"],
    "unit testing": ["unit tests", "pytest", "junit", "jest", "test driven development", "tdd"],
    "linux": ["unix", "bash scripting", "shell scripting"],
    # mechanical
    "autocad": ["auto cad", "auto-cad"],
    "solidworks": ["solid works"],
    "catia": ["catia v5"],
    "ansys": ["ansys workbench"],
    "gd&t": ["gdt", "geometric dimensioning and tolerancing", "geometric dimensioning"],
    "finite element analysis": ["fea"],
    "computational fluid dynamics": ["cfd"],
    "cad": ["computer aided design", "computer-aided design"],
    "cam": ["computer aided manufacturing"],
    "cnc": ["cnc machining", "cnc programming"],
    "hvac": ["heating ventilation and air conditioning"],
    "six sigma": ["6 sigma", "lean six sigma"],
    "creo": ["ptc creo", "pro/engineer", "proe"],
    # electrical
    "plc": ["programmable logic controller", "plc programming"],
    "scada": ["scada systems"],
    "vlsi": ["vlsi design"],
    "verilog": ["system verilog", "systemverilog"],
    "pcb design": ["pcb", "printed circuit board", "altium"],
    "embedded systems": ["embedded c", "firmware", "microcontroller", "microcontrollers"],
    "matlab": ["mat lab", "simulink"],
    # business
    "excel": ["microsoft excel", "ms excel", "advanced excel"],
    "power bi": ["powerbi", "microsoft power bi"],
    "tableau": ["tableau desktop"],
    "sap": ["sap erp"],
    "seo": ["search engine optimization", "search engine optimisation"],
    "google analytics": ["ga4", "google analytics 4"],
}

# Flattened alias -> canonical lookup
ALIASES: dict[str, str] = {}
for _canonical, _aliases in _ALIAS_GROUPS.items():
    ALIASES[_canonical] = _canonical
    for _alias in _aliases:
        ALIASES[_alias] = _canonical


# --------------------------------------------------------------------------- #
# 2. Related (transferable but NOT equivalent) skills
# --------------------------------------------------------------------------- #
def _symmetric(pairs: dict[tuple[str, str], str]) -> dict[str, dict[str, str]]:
    """Build a bidirectional related-skill graph from an edge list."""
    graph: dict[str, dict[str, str]] = {}
    for (left, right), relation in pairs.items():
        graph.setdefault(left, {})[right] = relation
        graph.setdefault(right, {})[left] = relation
    return graph


RELATED: dict[str, dict[str, str]] = _symmetric(
    {
        # --- relational databases: related, never equivalent ---
        ("postgresql", "mysql"): "both are relational SQL databases",
        ("postgresql", "sql server"): "both are relational SQL databases",
        ("mysql", "sql server"): "both are relational SQL databases",
        ("postgresql", "sql"): "PostgreSQL is a specific SQL database",
        ("mysql", "sql"): "MySQL is a specific SQL database",
        ("sql server", "sql"): "SQL Server is a specific SQL database",
        ("mongodb", "nosql"): "MongoDB is a NoSQL document database",
        ("elasticsearch", "nosql"): "Elasticsearch is a NoSQL search store",
        # --- cloud providers: transferable concepts ---
        ("aws", "azure"): "both are major public cloud platforms",
        ("aws", "gcp"): "both are major public cloud platforms",
        ("azure", "gcp"): "both are major public cloud platforms",
        # --- container / orchestration ---
        ("docker", "kubernetes"): "container tooling used together",
        ("kubernetes", "terraform"): "both are infrastructure/orchestration tooling",
        ("docker", "ci/cd"): "containers are commonly part of deployment pipelines",
        ("jenkins", "ci/cd"): "Jenkins is a CI/CD tool",
        ("github actions", "ci/cd"): "GitHub Actions is a CI/CD tool",
        ("jenkins", "github actions"): "both are CI/CD automation tools",
        # --- frontend frameworks: same category only ---
        ("react", "vue"): "both are component-based frontend frameworks",
        ("react", "angular"): "both are component-based frontend frameworks",
        ("angular", "vue"): "both are component-based frontend frameworks",
        ("react", "next.js"): "Next.js is a React framework",
        ("javascript", "typescript"): "TypeScript is a typed superset of JavaScript",
        # --- backend frameworks ---
        ("django", "flask"): "both are Python web frameworks",
        ("django", "fastapi"): "both are Python web frameworks",
        ("flask", "fastapi"): "both are Python web frameworks",
        ("express", "node.js"): "Express is the standard Node.js web framework",
        ("spring boot", "java"): "Spring Boot is the dominant Java backend framework",
        ("rest api", "graphql"): "both are API design approaches",
        ("rest api", "microservices"): "REST APIs are typical in microservice systems",
        # --- ML stack ---
        ("tensorflow", "pytorch"): "both are deep learning frameworks",
        ("machine learning", "deep learning"): "deep learning is a subfield of ML",
        ("machine learning", "scikit-learn"): "scikit-learn is a core ML library",
        ("deep learning", "natural language processing"): "NLP commonly uses deep learning",
        ("deep learning", "computer vision"): "computer vision commonly uses deep learning",
        ("machine learning", "large language models"): "LLMs are a branch of ML",
        ("natural language processing", "large language models"): "LLMs are used for NLP",
        ("pandas", "numpy"): "both are core Python data libraries",
        ("pandas", "sql"): "both are used for data manipulation",
        # --- mechanical ---
        ("autocad", "solidworks"): "both are CAD packages",
        ("autocad", "catia"): "both are CAD packages",
        ("solidworks", "catia"): "both are 3D CAD packages",
        ("solidworks", "creo"): "both are 3D CAD packages",
        ("autocad", "cad"): "AutoCAD is a CAD package",
        ("solidworks", "cad"): "SolidWorks is a CAD package",
        ("catia", "cad"): "CATIA is a CAD package",
        ("creo", "cad"): "Creo is a CAD package",
        ("ansys", "finite element analysis"): "ANSYS is an FEA tool",
        ("ansys", "computational fluid dynamics"): "ANSYS is used for CFD",
        ("finite element analysis", "computational fluid dynamics"): "both are simulation disciplines",
        ("cnc", "cam"): "CNC machining is driven by CAM software",
        ("six sigma", "lean manufacturing"): "both are process improvement methodologies",
        # --- electrical / embedded ---
        ("plc", "scada"): "both are industrial automation technologies",
        ("verilog", "vhdl"): "both are hardware description languages",
        ("verilog", "vlsi"): "Verilog is used in VLSI design",
        ("embedded systems", "c"): "embedded development is predominantly C",
        ("matlab", "simulink"): "Simulink is built on MATLAB",
        # --- business / analytics ---
        ("power bi", "tableau"): "both are BI visualisation tools",
        ("excel", "power bi"): "both are used for business reporting",
    }
)


# --------------------------------------------------------------------------- #
# 3. Domain vocabularies (for role-mismatch detection)
# --------------------------------------------------------------------------- #
DOMAIN_VOCABULARY: dict[str, set[str]] = {
    "software_engineering": {
        "software", "developer", "engineer", "programming", "python", "java",
        "javascript", "typescript", "react", "angular", "vue", "node.js",
        "django", "flask", "fastapi", "spring boot", "rest api", "graphql",
        "microservices", "backend", "frontend", "full stack", "fullstack",
        "api", "git", "agile", "unit testing", "code review", "sql",
        "postgresql", "mysql", "mongodb", "c++", "c#", "golang", "ruby",
        "web application", "mobile application", "android", "ios", "sdlc",
        "object oriented", "data structures", "algorithms", "debugging",
    },
    "data_science_ml": {
        "machine learning", "deep learning", "data science", "data scientist",
        "natural language processing", "computer vision", "tensorflow",
        "pytorch", "scikit-learn", "pandas", "numpy", "model training",
        "feature engineering", "statistics", "statistical", "regression",
        "classification", "clustering", "neural network", "large language models",
        "mlops", "data pipeline", "jupyter", "predictive model", "ai",
        "artificial intelligence", "model deployment", "dataset",
    },
    "devops_cloud": {
        "devops", "aws", "azure", "gcp", "kubernetes", "docker", "terraform",
        "ci/cd", "jenkins", "github actions", "infrastructure", "sre",
        "site reliability", "monitoring", "prometheus", "grafana", "ansible",
        "linux", "deployment", "cloud architecture", "scalability", "ec2", "s3",
        "iam", "load balancer", "helm", "observability",
    },
    "mechanical_engineering": {
        "mechanical", "autocad", "solidworks", "catia", "creo", "ansys", "cad",
        "cam", "gd&t", "finite element analysis", "computational fluid dynamics",
        "thermodynamics", "fluid mechanics", "manufacturing", "machining",
        "cnc", "tolerance", "tolerancing", "sheet metal", "casting", "welding",
        "hvac", "product design", "mechanical design", "bom", "bill of materials",
        "prototyping", "injection molding", "material science", "metallurgy",
        "six sigma", "lean manufacturing", "assembly", "drafting", "pneumatics",
        "hydraulics", "heat transfer", "stress analysis", "vibration",
        "automotive", "aerospace", "production planning", "quality control",
    },
    "electrical_engineering": {
        "electrical", "electronics", "circuit", "pcb design", "vlsi", "verilog",
        "vhdl", "embedded systems", "microcontroller", "plc", "scada", "matlab",
        "power systems", "transformer", "signal processing", "analog", "digital",
        "semiconductor", "fpga", "rf", "control systems", "instrumentation",
        "voltage", "electrical design", "wiring", "substation",
    },
    "civil_engineering": {
        "civil", "structural", "construction", "staad", "etabs", "revit",
        "surveying", "concrete", "reinforcement", "geotechnical", "soil",
        "bridge", "highway", "site engineer", "quantity surveying", "bim",
        "estimation", "foundation", "rcc", "steel structure",
    },
    "finance_accounting": {
        "finance", "accounting", "accountant", "audit", "taxation", "gaap",
        "ifrs", "financial modeling", "financial analysis", "budgeting",
        "forecasting", "reconciliation", "accounts payable", "accounts receivable",
        "ledger", "cpa", "ca", "invoice", "payroll", "tally", "quickbooks",
        "valuation", "equity research", "portfolio",
    },
    "marketing_sales": {
        "marketing", "sales", "seo", "sem", "campaign", "brand", "content marketing",
        "social media", "google analytics", "lead generation", "crm", "salesforce",
        "digital marketing", "advertising", "copywriting", "market research",
        "customer acquisition", "conversion", "b2b", "b2c", "hubspot",
    },
    "human_resources": {
        "human resources", "recruitment", "recruiting", "talent acquisition",
        "onboarding", "employee engagement", "payroll", "hris", "performance review",
        "sourcing", "interviewing", "hr policies", "compensation", "benefits",
        "workday", "employee relations",
    },
    "design_ux": {
        "ux", "ui", "user experience", "user interface", "figma", "sketch",
        "adobe xd", "wireframe", "prototype", "design system", "usability",
        "user research", "interaction design", "visual design", "photoshop",
        "illustrator", "typography",
    },
    "healthcare": {
        "nursing", "nurse", "patient", "clinical", "medical", "healthcare",
        "diagnosis", "treatment", "hospital", "pharmacy", "physician",
        "medical records", "hipaa", "phlebotomy", "vital signs", "caregiving",
    },
}

# Domains that are broadly compatible with one another. A JD in one and a resume
# in another is a *specialisation* difference, not a career mismatch.
COMPATIBLE_DOMAINS: list[set[str]] = [
    {"software_engineering", "data_science_ml", "devops_cloud"},
    {"electrical_engineering", "mechanical_engineering"},
    {"marketing_sales", "human_resources"},
    {"design_ux", "software_engineering"},
]

DOMAIN_LABELS: dict[str, str] = {
    "software_engineering": "Software Engineering",
    "data_science_ml": "Data Science / Machine Learning",
    "devops_cloud": "DevOps / Cloud Infrastructure",
    "mechanical_engineering": "Mechanical Engineering",
    "electrical_engineering": "Electrical / Electronics Engineering",
    "civil_engineering": "Civil Engineering",
    "finance_accounting": "Finance / Accounting",
    "marketing_sales": "Marketing / Sales",
    "human_resources": "Human Resources",
    "design_ux": "Design / UX",
    "healthcare": "Healthcare",
    "unknown": "Unclassified",
}


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
_PUNCT_RE = re.compile(r"[^a-z0-9+#./&\s-]")
_WS_RE = re.compile(r"\s+")
# A dot not followed by an alphanumeric is sentence punctuation, not part of a
# name such as "node.js" or "asp.net".
_TRAILING_DOT_RE = re.compile(r"\.(?![a-z0-9])")
# Strip qualifiers so "strong experience with Docker (3+ years)" -> "docker"
_NOISE_RE = re.compile(
    r"\b(experience|expertise|proficiency|proficient|knowledge|strong|solid|hands[- ]on|"
    r"working|advanced|basic|good|excellent|familiarity|familiar|with|in|of|and|the|a|an|"
    r"years?|yrs?)\b"
)
# Standalone quantities such as "3+" or "5". Anchored so that "c++" and version
# numbers inside a skill name are left alone.
_QUANTITY_RE = re.compile(r"(?<![a-z0-9])\d+\+*(?![a-z0-9])")


def normalize(term: str) -> str:
    """Lowercase, strip punctuation/noise, and collapse whitespace."""
    if not term:
        return ""
    text = term.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    # A dot is kept inside names like "node.js" and "asp.net", but a sentence
    # ending dot would otherwise stop "PostgreSQL." from matching "postgresql".
    text = _TRAILING_DOT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def canonical(term: str) -> str:
    """Map a raw skill string onto its canonical name.

    Falls back to the normalised term when the skill is not in the alias table,
    so unknown skills still match each other by string equality.
    """
    norm = normalize(term)
    if not norm:
        return ""
    if norm in ALIASES:
        return ALIASES[norm]

    # Try again after stripping filler words and quantities, e.g.
    # "3+ years of hands-on Docker experience" -> "docker".
    stripped = _QUANTITY_RE.sub(" ", norm)
    stripped = _WS_RE.sub(" ", _NOISE_RE.sub(" ", stripped)).strip()
    # Drop leftover separator punctuation, while preserving names like "c++".
    stripped = stripped.strip(" -_/,.")
    if stripped and stripped in ALIASES:
        return ALIASES[stripped]
    return stripped or norm


def canonical_set(terms: Iterable[str]) -> set[str]:
    """Canonicalise a collection of skills, dropping empties."""
    return {c for c in (canonical(t) for t in terms) if c}


def relation_between(left: str, right: str) -> str | None:
    """Return the relation description if two canonical skills are related."""
    return RELATED.get(left, {}).get(right)


def classify_domain(text: str, skills: Iterable[str] = ()) -> tuple[str, dict[str, float]]:
    """Classify free text (plus optional skill list) into a professional domain.

    Returns ``(top_domain, scores)`` where scores are normalised 0-1 shares of
    the total vocabulary hits. Returns ``"unknown"`` when there is too little
    signal to make a call.
    """
    haystack = " " + normalize(text) + " "
    skill_terms = canonical_set(skills)

    raw: dict[str, float] = {}
    for domain, vocabulary in DOMAIN_VOCABULARY.items():
        hits = 0.0
        for term in vocabulary:
            norm_term = normalize(term)
            if not norm_term:
                continue
            if f" {norm_term} " in haystack:
                hits += 1.0
            if norm_term in skill_terms:
                # An explicitly extracted skill is stronger evidence than prose.
                hits += 1.5
        raw[domain] = hits

    total = sum(raw.values())
    if total < 3:
        return "unknown", {d: 0.0 for d in raw}

    scores = {d: v / total for d, v in raw.items()}
    top = max(scores, key=lambda d: scores[d])
    return top, scores


def domains_compatible(left: str, right: str) -> bool:
    """True when two domains are close enough to not count as a role mismatch."""
    if left == right:
        return True
    if "unknown" in (left, right):
        return True  # not enough signal - do not accuse of a mismatch
    return any({left, right} <= group for group in COMPATIBLE_DOMAINS)


def domain_label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain.replace("_", " ").title())


# --------------------------------------------------------------------------- #
# Display formatting
# --------------------------------------------------------------------------- #
# Canonical names are lowercase for matching; these are their presentation forms.
_DISPLAY_OVERRIDES: dict[str, str] = {
    "aws": "AWS", "gcp": "GCP", "sql": "SQL", "nosql": "NoSQL",
    "ci/cd": "CI/CD", "rest api": "REST API", "graphql": "GraphQL",
    "html": "HTML", "css": "CSS", "php": "PHP", "gd&t": "GD&T",
    "cad": "CAD", "cam": "CAM", "cnc": "CNC", "plc": "PLC", "scada": "SCADA",
    "vlsi": "VLSI", "vhdl": "VHDL", "hvac": "HVAC", "pcb design": "PCB Design",
    "sap": "SAP", "seo": "SEO", "crm": "CRM", "erp": "ERP",
    "javascript": "JavaScript", "typescript": "TypeScript",
    "node.js": "Node.js", "next.js": "Next.js", "vue": "Vue",
    "asp.net": "ASP.NET", "c++": "C++", "c#": "C#", "golang": "Go",
    "postgresql": "PostgreSQL", "mysql": "MySQL", "mongodb": "MongoDB",
    "sql server": "SQL Server", "sqlite": "SQLite", "dynamodb": "DynamoDB",
    "elasticsearch": "Elasticsearch", "redis": "Redis",
    "machine learning": "Machine Learning", "deep learning": "Deep Learning",
    "natural language processing": "Natural Language Processing (NLP)",
    "computer vision": "Computer Vision",
    "large language models": "Large Language Models (LLMs)",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch",
    "scikit-learn": "scikit-learn", "numpy": "NumPy", "pandas": "pandas",
    "github actions": "GitHub Actions", "matlab": "MATLAB",
    "autocad": "AutoCAD", "solidworks": "SolidWorks", "catia": "CATIA",
    "ansys": "ANSYS", "creo": "Creo",
    "finite element analysis": "Finite Element Analysis (FEA)",
    "computational fluid dynamics": "Computational Fluid Dynamics (CFD)",
    "power bi": "Power BI", "ios": "iOS",
    "unit testing": "Unit Testing", "six sigma": "Six Sigma",
    "embedded systems": "Embedded Systems",
}


def display_name(skill: str) -> str:
    """Presentation form of a skill for user-facing output.

    Matching works on lowercase canonical names; this turns "ci/cd" back into
    "CI/CD" and "rest api" into "REST API" for the report.
    """
    if not skill:
        return ""
    raw = skill.strip()
    canon = canonical(raw)
    if canon in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[canon]
    if raw.lower() in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[raw.lower()]
    # Preserve wording the user already capitalised themselves.
    if raw != raw.lower():
        return raw
    return " ".join(word.capitalize() if word.islower() else word for word in raw.split())
