# 🤖 ResumeMatch AI — Discord Resume ↔ JD Matching Bot

A Discord chatbot that compares one or more resumes against a job description and
returns a recruiter-grade analysis. **The entire experience lives inside Discord** —
there is no web frontend, no dashboard and no separate UI.

Upload a JD, upload resumes, get ranked candidates with explainable scores.

---

## 1. Project Overview

A recruiter drops a job description and a handful of resumes into a Discord
channel. The bot works out which file is which, extracts structured information
from each, scores every resume against the JD with a transparent weighted
engine, and posts a ranked comparison plus a full per-candidate breakdown.

The design principle throughout: **the LLM interprets, Python decides.**
Semantic understanding (what the JD asks for, what a resume demonstrates) is the
model's job. Every number — the ATS score, the skill match, the ranking — is
computed in Python from validated structures, so it is reproducible, auditable
and identical on every run.

---

## 2. Features

### The five required outputs, for every resume

| S.NO | Output | What you get |
|---|--------|--------------|
| 1 | 🎯 **ATS Score** | 0–100 from a six-component weighted engine, with the full breakdown shown |
| 2 | 🧩 **Skill Set Match** | Strong / partial / missing, required skills weighted 3× over preferred |
| 3 | ❌ **Missing Areas** | Prioritised HIGH / MEDIUM / LOW, spanning skills, tools, experience, domain knowledge, certifications, responsibilities and project evidence |
| 4 | 📈 **Job Alignment** | Six-dimension breakdown plus a plain-language explanation with strong/weak points |
| 5 | 📚 **Recommended Courses** | Real course names and providers, each justified by a specific gap — never invented URLs |

### Plus

- 🧠 **Learning Roadmap** — prioritised list of what to actually study, with topics
- ⚠️ **Role Mismatch Detection** — a Mechanical Engineering JD scores a Software
  Engineering resume at **8/100**, not 80, no matter how many impressive keywords it contains
- 🏆 **Multi-resume ranking** — one JD, many resumes, a comparison table and a best-fit pick
- 🟢🟡🔴 **Confidence indicator** — driven by text quality, JD completeness and parse warnings
- 🔍 **Automatic document classification** — you never have to say "this is the JD"
- 💡 **Resume improvement suggestions** — the top practical fixes to the document itself
- 🧾 **Evidence-based language** — "No evidence of Docker was found in the resume",
  never "the candidate doesn't know Docker"

---

## 3. Architecture

```
Discord message
      │
      ▼
┌──────────────────┐   discord_bot.py — events, uploads, message splitting
│   Discord layer  │   handlers.py    — conversation logic (no Discord types)
│                  │   session.py     — per-(channel,user) state machine
└────────┬─────────┘
         ▼
┌──────────────────┐   pdf_parser / docx_parser / text_parser
│   Parsing layer  │   loader.py      — routing, size limits, error handling
│                  │   classifier.py  — is this a JD or a resume?
└────────┬─────────┘
         ▼
┌──────────────────┐   client.py   — provider-agnostic, JSON-only, retries + repair
│    AI layer      │   prompts.py  — extraction and narrative prompts
│   (optional)     │   schemas.py  — the JSON contract
└────────┬─────────┘
         │  ⇅ falls back to heuristic_extractor.py if unavailable
         ▼
┌──────────────────┐   skill_taxonomy.py — synonyms, related-skill graph, domains
│  Analysis layer  │   skill_matcher.py  — exact / partial / missing
│  (deterministic) │   ats_scorer.py     — weighted scoring engine
│                  │   role_matcher.py   — domain mismatch detection
│                  │   recommender.py    — gaps, courses, roadmap
│                  │   analyzer.py       — orchestration + confidence
└────────┬─────────┘
         ▼
┌──────────────────┐   discord_format.py — markdown, bars, 2000-char packing
│ Formatting layer │
└──────────────────┘
```

### What the LLM does vs. what Python does

| LLM (semantic) | Python (deterministic) |
|----------------|------------------------|
| Extracting JD requirements and splitting required vs preferred | All scoring and weighting |
| Extracting structured resume information | Skill matching and synonym resolution |
| Writing explanations and strong/weak points | Candidate ranking |
| Spotting gaps that keyword matching misses | Session state, validation, formatting |

If the LLM is missing, misconfigured, rate-limited or returns unusable JSON, the
deterministic extractor takes over and the analysis still completes — the
confidence indicator drops to say so. **The demo cannot be broken by an API outage.**

---

## 4. Tech Stack

| Concern | Choice |
|---------|--------|
| Language | Python 3.11+ (developed and tested on 3.13) |
| Chat platform | `discord.py` 2.5.2 |
| PDF parsing | PyMuPDF (`fitz`) |
| DOCX parsing | `python-docx` |
| Validation | Pydantic v2 |
| Config | `pydantic-settings` + `python-dotenv` |
| HTTP | `httpx` (async) |
| Testing | `pytest` + `pytest-asyncio` |

No FastAPI, no database, no frontend — none of them earn their place in this MVP.

---

## 5. Setup

### Prerequisites

- Python 3.11 or newer (`python --version`)
- A Discord account

### Install

```powershell
cd C:\Users\varshitha\Downloads\Hashira

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

macOS / Linux: use `source .venv/bin/activate` instead.

---

## 6. Creating and Connecting the Discord Bot

1. **Create the application**
   Go to <https://discord.com/developers/applications> → **New Application** →
   name it `ResumeMatch AI` → **Create**.

2. **Create the bot user**
   Left sidebar → **Bot** → **Add Bot** → **Yes, do it**.

3. **Copy the token**
   Still on the **Bot** page → **Reset Token** → **Copy**.
   Paste it into `.env` as `DISCORD_BOT_TOKEN`. Treat it like a password.

4. **Enable the Message Content Intent** ⚠️ *required*
   On the **Bot** page, scroll to **Privileged Gateway Intents** and switch
   **MESSAGE CONTENT INTENT** to **ON**, then **Save Changes**.
   Without this the bot cannot read uploaded files and will appear to ignore you.

5. **Generate the invite link**
   Left sidebar → **OAuth2** → **URL Generator**.
   - **Scopes:** `bot`, `applications.commands`
   - **Bot Permissions:** `Send Messages`, `Read Message History`,
     `Attach Files`, `Embed Links`, `Use Slash Commands`
   Copy the generated URL at the bottom.

6. **Invite the bot**
   Paste that URL into your browser, pick a server you administer, **Authorize**.

7. **Run it** (see below). The bot appears online, and `/start` works in any
   channel it can see, or in a DM with the bot.

---

## 7. Environment Variables

Copy `.env.example` to `.env` and fill it in:

```powershell
copy .env.example .env
```

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DISCORD_BOT_TOKEN` | ✅ yes | — | Your Discord bot token |
| `LLM_API_KEY` | unless `DEMO_MODE=true` | — | API key for your LLM provider |
| `LLM_MODEL` | no | `claude-sonnet-5` | Model name |
| `LLM_PROVIDER` | no | `anthropic` | `anthropic` or `openai` (wire format) |
| `LLM_BASE_URL` | no | `https://api.anthropic.com/v1` | Provider base URL |
| `DEMO_MODE` | no | `false` | `true` runs the deterministic engine with zero API calls |
| `MAX_FILES_PER_SESSION` | no | `10` | Upload cap per session |
| `MAX_FILE_SIZE_MB` | no | `10` | Per-file size cap |
| `MAX_DOCUMENT_CHARS` | no | `20000` | Truncation limit before prompting |
| `SESSION_TTL_MINUTES` | no | `60` | Session expiry |
| `LOG_LEVEL` | no | `INFO` | Logging verbosity |

### Switching LLM provider

The client is provider-agnostic — changing provider is a `.env` edit, not a code change.

```ini
# Anthropic (default)
LLM_PROVIDER=anthropic
LLM_BASE_URL=https://api.anthropic.com/v1
LLM_MODEL=claude-sonnet-5

# OpenAI
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Local Ollama — no API cost at all
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1
LLM_API_KEY=ollama
```

`LLM_PROVIDER=openai` works with any OpenAI-compatible gateway (Groq, Together,
OpenRouter, LM Studio, vLLM …).

### Security

- `.env` is in `.gitignore` and is never committed.
- API keys are never logged — `Settings.safe_summary()` prints `set` / `missing` only.
- Resume text is held in memory for the session only, never written to disk, and
  is excluded from the JSON export.

---

## 8. Running Locally

```powershell
.venv\Scripts\activate
python run.py
```

Expected output:

```
ResumeMatch AI starting up…
Configuration: {'llm_provider': 'anthropic', ..., 'llm_api_key': 'set', ...}
Synced 6 slash command(s).
Logged in as ResumeMatch AI#1234 (id=…)
Connected to 1 guild(s).
```

### Running without Discord

`demo.py` runs the identical pipeline in the terminal and prints exactly what the
bot would post — useful for verifying everything before a demo, or as a fallback.

```powershell
# One JD vs four very different resumes
python demo.py

# The recruiter's headline test: mechanical JD + software resume
python demo.py --scenario mismatch

# Your own files
python demo.py --jd path\to\jd.pdf --resume a.pdf --resume b.docx

# Export structured results as JSON
python demo.py --json results.json
```

Scenarios: `compare`, `strong`, `weak`, `mismatch`, `ml`, `backend`, `mechanical`.

### Sample files for the demo

`samples/` contains ready-made PDF and DOCX files you can drag straight into
Discord — no need to find real resumes:

| File | Use in the demo |
|------|-----------------|
| `job_description_software_engineer.pdf` | The main JD |
| `resume_priya_raghavan_strong.pdf` | Strong match (~89/100) |
| `resume_neha_kulkarni_fullstack.docx` | Good match, DOCX format (~74/100) |
| `resume_arjun_mehta_junior.pdf` | Weak match (~32/100) |
| `resume_rohit_deshmukh_mechanical.pdf` | Role mismatch (~11/100) |
| `job_description_mechanical_engineer.pdf` | For the mismatch test |
| `job_description_ml_engineer.pdf` | Adjacent-role test |
| `job_description_backend_engineer.txt` | TXT format test |

Regenerate them any time with `python make_samples.py`.

---

## 9. Running Tests

```powershell
.venv\Scripts\activate
python -m pytest              # all 331 tests
python -m pytest -v           # verbose
python -m pytest tests/test_scoring.py    # one file
```

Tests never call a real LLM API: the suite forces `DEMO_MODE=true`, and the tests
that exercise the LLM path mock the HTTP transport.

**Coverage:**

| File | Tests | Area |
|------|-------|------|
| `test_parsers.py` | 36 | PDF / DOCX / TXT, corrupt, empty, scanned, oversized, unsupported, JD-vs-resume classification |
| `test_extraction.py` | 33 | Resume and JD structured extraction, evidence-only skill detection |
| `test_matching.py` | 48 | Synonyms, partial matches, unrelated-skill rejection, required vs preferred |
| `test_scoring.py` | 28 | Weighted ATS engine, all six components, role mismatch, the recruiter's scenarios |
| `test_analysis.py` | 43 | The five required outputs, alignment gaps, course links, ranking, formatting |
| `test_session.py` | 51 | State machine, session isolation, upload/reject rules, reset |
| `test_llm.py` | 36 | JSON repair, wire formats, auth/timeout/rate-limit handling, fallback |
| `test_discord_bot.py` | 33 | Full flows through the bot with stubbed Discord objects, small talk |
| `test_qa_hardening.py` | 23 | Concurrency guard, command safety, session sweep, LLM circuit breaker |

---

## 10. Example Interaction

**You:** `/start`

> 👋 **Welcome to ResumeMatch AI!**
> I compare resumes against a Job Description.
> Please provide: 1️⃣ a Job Description  2️⃣ one or more Resumes

**You:** *uploads* `software_jd.pdf`

> ✅ **Job Description received** — `software_jd.pdf` (1,901 characters read).
> **Now upload one or more resumes.** You can attach several at once.

**You:** *uploads* `resume1.pdf` `resume2.pdf` `resume3.pdf`

> 📄 `resume1.pdf` received as a **resume** (2,576 characters read).
> 📄 `resume2.pdf` received as a **resume** (957 characters read).
> 📄 `resume3.pdf` received as a **resume** (2,221 characters read).
> 🔍 **Analyzing 3 resumes against the job description…**

> ```
> #  CANDIDATE         ATS   SKILL   ALIGN  VERDICT
> --------------------------------------------------------
> 1  Priya Raghavan     89     94%     90%  Strong fit
> 2  Arjun Mehta        32     18%     21%  Weak fit
> 3  Rohit Deshmukh     11      0%      9%  Role mismatch
> ```
> 🏆 **Best aligned candidate: Priya Raghavan** — ATS 89/100, 94% skill match, 90% alignment.
> ⚠️ **Role mismatch detected for:** Rohit Deshmukh

…followed by a full per-candidate breakdown: ATS bars, skill match, missing
areas, alignment, courses and roadmap.

### Commands

| Command | Does |
|---------|------|
| `/start` | Begin a new session |
| `/analyze` | Force analysis with whatever is uploaded |
| `/status` | Show what the bot currently holds |
| `/reset` | Clear the JD and all resumes |
| `/help` | Full help |
| `/jd <text>` | Paste a JD as text instead of uploading |

All of these also work typed as plain text or with a `!` prefix, so the bot is
usable even before slash commands finish propagating.

---

## 11. Supported Files

| Format | JD | Resume | Notes |
|--------|----|--------|-------|
| `.pdf` | ✅ | ✅ | Text-based PDFs. Scanned/image-only PDFs are detected and reported |
| `.docx` | ✅ | ✅ | Paragraphs *and* table cells are read |
| `.txt` / `.md` | ✅ | ✅ | UTF-8, UTF-16 and CP1252 all handled |
| Pasted text | ✅ | — | Via `/jd` or by pasting a long message |
| `.doc` | ❌ | ❌ | Legacy format — the bot tells you to save as `.docx` |
| Images | ❌ | ❌ | No text layer; the bot explains why |

Limits: 10 MB per file, 10 files per session (both configurable).

---

## 12. Scoring Methodology

### ATS score — six weighted components

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Required skill match | **25%** | The strongest predictor of fit; missing must-haves should dominate |
| Keyword match | **20%** | Mirrors how real ATS software filters, and captures JD vocabulary beyond the skill list |
| Experience match | **20%** | Seniority and years gate most shortlists |
| Project / responsibility relevance | **15%** | Evidence the skills were actually applied, not just listed |
| Education / certification | **10%** | Usually a gate, rarely a differentiator |
| Resume structure / parseability | **10%** | A genuine ATS signal: contact details, recognisable sections, usable length |

Each component is scored 0–100 independently, then combined. **No score is ever
asked of the LLM.**

### Semantic skill matching

Three outcomes, deliberately conservative:

| Status | Meaning | Credit |
|--------|---------|--------|
| ✅ **Exact** | The resume names the skill or a true synonym — `JS` = `JavaScript`, `Postgres` = `PostgreSQL`, `K8s` = `Kubernetes` | 1.0 |
| 🟡 **Partial** | A genuinely related skill from a curated graph — MySQL for a PostgreSQL requirement, Azure for AWS, Django for Flask | 0.5 |
| ❌ **Missing** | No evidence found in the document | 0.0 |

Unrelated skills are **never** conflated: Python does not satisfy Java, React does
not satisfy Django. Required skills carry **3×** the weight of preferred skills, so
nice-to-haves cannot paper over a missing must-have.

### Role mismatch detection

Both documents are classified into a professional domain (software, data/ML,
DevOps, mechanical, electrical, civil, finance, marketing, HR, design, healthcare)
using curated vocabularies. When the JD's domain and the resume's domain are
incompatible, a penalty **and a hard ceiling** are applied after weighting:

| Severity | Multiplier | Ceiling |
|----------|------------|---------|
| HIGH | ×0.30 | 30/100 |
| MEDIUM | ×0.55 | 55/100 |
| LOW | ×0.85 | — |

Adjacent domains (software ↔ data/ML ↔ DevOps) are treated as **compatible**: a
full-stack resume against a backend JD is a specialisation difference, not a
career mismatch, and is scored normally. When either document is too sparse to
classify confidently, **no mismatch is claimed** — the bot does not accuse on thin evidence.

*Verified result:* mechanical JD + strong software resume → **8/100**, mismatch
flagged HIGH. The same resume against its matching software JD → **89/100**.

### Alignment score

Weighted separately from ATS: technical skills 30%, experience 20%, projects 15%,
responsibilities 15%, keywords 10%, education 10% — then the same mismatch penalty.

### Confidence

Starts at 100 and is reduced for short extracted text, parse warnings, missing
resume structure, no identifiable skills, a vague JD, and running without LLM
assistance. ≥75 → 🟢 High, ≥50 → 🟡 Medium, else 🔴 Low.

---

## 13. Limitations

- **Scanned PDFs are not OCR'd.** They are detected and reported with a clear
  message rather than silently scored. Adding OCR would mean a Tesseract
  dependency that complicates Windows setup for little demo value.
- **Sessions are in-memory.** A restart clears them. Deliberate for an MVP —
  swapping in Redis means replacing one class (`SessionStore`).
- **The skill taxonomy is hand-curated.** It covers software, data/ML, DevOps,
  mechanical, electrical, civil, finance, marketing, HR, design and healthcare
  well; a niche domain will fall back to literal string matching, which is
  conservative rather than wrong.
- **Years of experience** are read from explicit statements or inferred from date
  ranges; overlapping roles can inflate the inferred figure.
- **English only.**
- **The heuristic extractor is deliberately literal** — it will not infer that a
  "Backend Engineer" knows Docker. That is a feature (no invented skills), but it
  scores slightly lower than the LLM path on unusually formatted resumes.
- **No persistence or audit trail** beyond the optional JSON export.

---

## 14. Future Improvements

1. **OCR fallback** (Tesseract / PaddleOCR) for scanned resumes.
2. **Embedding-based similarity** to complement the curated related-skill graph,
   with the deterministic scores kept as the source of truth.
3. **Redis-backed sessions** for multi-instance deployment.
4. **Per-JD weight profiles** — let a recruiter say experience matters more than
   education for a given role.
5. **Bulk mode** — a ZIP of 50 resumes, results as a downloadable CSV.
6. **Interview question generation** targeted at each candidate's specific gaps.
7. **Bias auditing** — flag when scores correlate with name, gender or location
   signals rather than skills.
8. **A `/why` command** to interrogate any single score component interactively.

---

## 15. Project Structure

```
Hashira/
├── app/
│   ├── main.py                     # logging, preflight checks, startup
│   ├── config.py                   # env-driven settings, never logs secrets
│   ├── bot/
│   │   ├── discord_bot.py          # Discord events + slash commands
│   │   ├── handlers.py             # conversation logic (Discord-free)
│   │   └── session.py              # state machine + session store
│   ├── parsers/
│   │   ├── loader.py               # routing, size limits, error handling
│   │   ├── pdf_parser.py           # PyMuPDF
│   │   ├── docx_parser.py          # python-docx
│   │   ├── text_parser.py          # TXT / pasted text
│   │   └── classifier.py           # JD vs resume detection
│   ├── analysis/
│   │   ├── skill_taxonomy.py       # synonyms, related graph, domains
│   │   ├── skill_matcher.py        # exact / partial / missing
│   │   ├── ats_scorer.py           # weighted scoring engine
│   │   ├── role_matcher.py         # domain mismatch detection
│   │   ├── recommender.py          # gaps, courses, roadmap
│   │   ├── heuristic_extractor.py  # LLM-free extraction
│   │   └── analyzer.py             # orchestration + confidence
│   ├── ai/
│   │   ├── client.py               # provider-agnostic, JSON-only
│   │   ├── prompts.py              # extraction + narrative prompts
│   │   └── schemas.py              # the JSON contract
│   ├── formatting/
│   │   └── discord_format.py       # markdown, bars, message packing
│   └── models/
│       └── models.py               # all Pydantic models
├── tests/                          # 331 tests
│   ├── fixtures/                   # 8 realistic JD/resume fixtures
│   ├── conftest.py
│   ├── test_parsers.py
│   ├── test_extraction.py
│   ├── test_matching.py
│   ├── test_scoring.py
│   ├── test_analysis.py
│   ├── test_session.py
│   ├── test_llm.py
│   └── test_discord_bot.py
├── samples/                        # ready-to-upload PDF/DOCX demo files
├── make_samples.py                 # regenerates samples/ from the fixtures
├── demo.py                         # terminal demo, no Discord needed
├── run.py                          # entry point
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

---

## 16. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Bot is online but ignores uploads | **Message Content Intent** is off — enable it in the developer portal (step 4) |
| `Discord rejected the bot token` | You copied the client secret or application ID. Use **Bot → Reset Token** |
| Slash commands don't appear | Re-invite with the `applications.commands` scope; global sync can take up to an hour. The `!` and plain-text commands work immediately |
| `ModuleNotFoundError: audioop` | `discord.py < 2.5.2` on Python 3.13 — `pip install -r requirements.txt` |
| `The LLM API rejected the credentials` | Check `LLM_API_KEY`, or set `DEMO_MODE=true` to run without an LLM |
| Analysis feels slow | Each resume is one LLM call plus one shared narrative call. Set `DEMO_MODE=true` for an instant demo |
| Emoji look wrong in the console | Cosmetic Windows code-page issue; Discord output is unaffected |
