"""Run an analysis from the command line - no Discord connection required.

This prints exactly what the bot would post into Discord, which makes it useful
for verifying the pipeline, for CI, and as a fallback if the Discord token is
not available during a demonstration.

Examples (Windows):

    # Built-in demo: one JD against three very different resumes
    python demo.py

    # The recruiter's headline test: mechanical JD vs software resume
    python demo.py --scenario mismatch

    # Your own files
    python demo.py --jd path\\to\\jd.pdf --resume a.pdf --resume b.docx

    # Export structured results
    python demo.py --json results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

from app.analysis.analyzer import run_analysis
from app.formatting import discord_format
from app.main import configure_logging
from app.models.models import ComparisonReport
from app.parsers import loader

FIXTURES = pathlib.Path(__file__).parent / "tests" / "fixtures"

SCENARIOS: dict[str, tuple[str, list[str]]] = {
    "compare": (
        "software_jd.txt",
        [
            "strong_software_resume.txt",
            "fullstack_resume.txt",
            "weak_software_resume.txt",
            "mechanical_resume.txt",
        ],
    ),
    "strong": ("software_jd.txt", ["strong_software_resume.txt"]),
    "weak": ("software_jd.txt", ["weak_software_resume.txt"]),
    "mismatch": ("mechanical_jd.txt", ["strong_software_resume.txt"]),
    "ml": ("ml_engineer_jd.txt", ["strong_software_resume.txt"]),
    "backend": ("backend_jd.txt", ["fullstack_resume.txt"]),
    "mechanical": ("mechanical_jd.txt", ["mechanical_resume.txt"]),
}


def load(path: pathlib.Path):
    """Read and parse one document from disk."""
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    return loader.parse_document(path.read_bytes(), path.name)


def report_to_dict(report: ComparisonReport) -> dict:
    """Serialise a report for the --json export."""
    return {
        "job_title": report.job_title,
        "candidate_count": len(report.results),
        "best_candidate": report.best.candidate_name if report.best else None,
        "ranking": [
            {
                "rank": index,
                "candidate_name": result.candidate_name,
                "source_file": result.source_filename,
                "ats_score": result.ats_score,
                "skill_match_score": result.skill_match_score,
                "alignment_score": result.alignment_score,
                "verdict": result.verdict,
                "role_mismatch": result.role_mismatch.detected,
                "confidence": result.confidence.value,
            }
            for index, result in enumerate(report.ranked, start=1)
        ],
        "analyses": [
            json.loads(result.model_dump_json()) for result in report.ranked
        ],
    }


async def run(args: argparse.Namespace) -> int:
    if args.jd:
        jd_document = load(pathlib.Path(args.jd))
        if not args.resume:
            raise SystemExit("Provide at least one --resume alongside --jd.")
        resume_documents = [load(pathlib.Path(r)) for r in args.resume]
    else:
        jd_name, resume_names = SCENARIOS[args.scenario]
        print(f"Scenario: {args.scenario}  (JD: {jd_name})\n")
        jd_document = load(FIXTURES / jd_name)
        resume_documents = [load(FIXTURES / name) for name in resume_names]

    async def progress(message: str) -> None:
        print(f"  {message}")

    report = await run_analysis(jd_document, resume_documents, progress=progress)
    print()

    if args.json:
        output = pathlib.Path(args.json)
        output.write_text(
            json.dumps(report_to_dict(report), indent=2, default=str), encoding="utf-8"
        )
        print(f"Structured results written to {output}\n")

    messages = discord_format.format_full_report(report)
    for index, message in enumerate(messages, start=1):
        print(f"┌── Discord message {index}/{len(messages)}  ({len(message)} chars)")
        print(message)
        print("└" + "─" * 60 + "\n")

    longest = max(len(m) for m in messages)
    print(
        f"{len(messages)} message(s); longest {longest} characters "
        f"(Discord limit is {discord_format.DISCORD_LIMIT})."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a ResumeMatch AI analysis in the terminal."
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="compare",
        help="Which built-in demo scenario to run (default: compare).",
    )
    parser.add_argument("--jd", help="Path to your own job description file.")
    parser.add_argument(
        "--resume",
        action="append",
        help="Path to a resume file. Repeat the flag for multiple resumes.",
    )
    parser.add_argument("--json", help="Also write structured results to this path.")
    args = parser.parse_args()

    configure_logging("WARNING")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
