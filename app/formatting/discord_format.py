"""Rendering of analysis results into Discord-friendly markdown.

Discord caps a message at 2000 characters, so everything here is built as a list
of self-contained blocks which are then packed into messages that never exceed
the limit and never split a code fence down the middle.
"""

from __future__ import annotations

from app.analysis import skill_taxonomy as tax
from app.models.models import (
    AnalysisResult,
    ComparisonReport,
    MissingArea,
    Priority,
    SkillImportance,
    SkillMatchResult,
)

DISCORD_LIMIT = 2000
_SAFE_LIMIT = 1900          # headroom for Discord's own formatting
_BAR_WIDTH = 12

_PRIORITY_ORDER = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
_PRIORITY_LABEL = {
    Priority.HIGH: "🔴 HIGH PRIORITY",
    Priority.MEDIUM: "🟠 MEDIUM PRIORITY",
    Priority.LOW: "🟡 LOW PRIORITY",
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def bar(value: float, width: int = _BAR_WIDTH) -> str:
    """A simple filled/empty block bar for a 0-100 value."""
    value = max(0.0, min(float(value), 100.0))
    filled = int(round(value / 100.0 * width))
    return "█" * filled + "░" * (width - filled)


def _skill_names(matches: list[SkillMatchResult], limit: int = 14) -> str:
    """Comma-separated display names, marking required skills with an asterisk."""
    names: list[str] = []
    for match in matches[:limit]:
        name = tax.display_name(match.skill)
        if match.importance == SkillImportance.REQUIRED:
            name = f"**{name}**"
        names.append(name)
    suffix = f" _+{len(matches) - limit} more_" if len(matches) > limit else ""
    return ", ".join(names) + suffix


def _score_emoji(score: float) -> str:
    if score >= 80:
        return "🟢"
    if score >= 60:
        return "🟡"
    if score >= 40:
        return "🟠"
    return "🔴"


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #
def format_comparison(report: ComparisonReport) -> list[str]:
    """The multi-candidate ranking table. Returns a list of blocks."""
    ranked = report.ranked
    if not ranked:
        return ["No results to compare."]

    blocks: list[str] = []
    header = (
        f"# 🏆 CANDIDATE COMPARISON\n"
        f"**Role:** {report.job_title}\n"
        f"**Candidates analysed:** {len(ranked)}\n"
    )
    blocks.append(header)

    # Fixed-width table inside a code block so Discord keeps the alignment.
    name_width = max(12, min(22, max(len(r.candidate_name) for r in ranked)))
    lines = [
        f"{'#':<3}{'CANDIDATE':<{name_width}}  {'ATS':>5}  {'SKILL':>6}  {'ALIGN':>6}  VERDICT",
        f"{'-' * (3 + name_width + 2 + 5 + 2 + 6 + 2 + 6 + 2 + 14)}",
    ]
    for index, result in enumerate(ranked, start=1):
        name = result.candidate_name
        if len(name) > name_width:
            name = name[: name_width - 1] + "…"
        lines.append(
            f"{index:<3}{name:<{name_width}}  "
            f"{result.ats_score:>5.0f}  "
            f"{result.skill_match_score:>5.0f}%  "
            f"{result.alignment_score:>5.0f}%  "
            f"{result.verdict}"
        )
    blocks.append("```\n" + "\n".join(lines) + "\n```")

    best = ranked[0]
    if best.role_mismatch.detected:
        blocks.append(
            "⚠️ **No strong match found.** Every candidate analysed shows a role "
            "or domain mismatch against this job description."
        )
    else:
        blocks.append(
            f"🏆 **Best aligned candidate: {best.candidate_name}** "
            f"— ATS {best.ats_score:.0f}/100, "
            f"{best.skill_match_score:.0f}% skill match, "
            f"{best.alignment_score:.0f}% alignment.\n"
            f"{best.confidence.emoji} Analysis confidence: {best.confidence.value.title()}"
        )

    mismatched = [r for r in ranked if r.role_mismatch.detected]
    if mismatched and len(mismatched) < len(ranked):
        names = ", ".join(r.candidate_name for r in mismatched)
        blocks.append(f"⚠️ **Role mismatch detected for:** {names}")

    return blocks


# --------------------------------------------------------------------------- #
# Individual analysis
# --------------------------------------------------------------------------- #
def format_result(result: AnalysisResult, index: int | None = None) -> list[str]:
    """Render one candidate's full analysis as a list of blocks."""
    blocks: list[str] = []
    prefix = f"{index}. " if index is not None else ""

    # --- header ---
    blocks.append(
        f"# 📋 {prefix}{result.candidate_name}\n"
        f"`{result.source_filename}` • **{result.job_title}**\n"
        f"{result.confidence.emoji} **{result.confidence.value.title()} confidence** "
        f"— {result.confidence_reason}"
    )

    # --- role mismatch banner (first, it reframes everything below) ---
    if result.role_mismatch.detected:
        blocks.append(_format_mismatch(result))

    # --- 1. ATS score ---
    blocks.append(_format_ats(result))

    # --- 2. skill set match ---
    blocks.append(_format_skills(result))

    # --- 3. missing areas ---
    blocks.append(_format_missing_areas(result.missing_areas))

    # --- 4. alignment ---
    blocks.append(_format_alignment(result))

    # --- 5. courses ---
    blocks.append(_format_courses(result))

    # --- learning roadmap ---
    roadmap = _format_roadmap(result)
    if roadmap:
        blocks.append(roadmap)

    # --- resume improvements (secondary) ---
    if result.improvement_suggestions:
        lines = ["## 💡 Top Things To Improve In The Resume"]
        for number, suggestion in enumerate(result.improvement_suggestions, start=1):
            lines.append(f"{number}. {suggestion}")
        blocks.append("\n".join(lines))

    if result.warnings:
        blocks.append(
            "⚠️ **Document notes:** " + " ".join(f"_{w}_" for w in result.warnings[:3])
        )

    return blocks


def _format_mismatch(result: AnalysisResult) -> str:
    mismatch = result.role_mismatch
    lines = [
        "## ⚠️ ROLE MISMATCH DETECTED",
        f"**JD role:** {mismatch.jd_role}",
        f"**Resume profile:** {mismatch.resume_profile} "
        f"({tax.domain_label(mismatch.resume_domain)})",
        f"**Severity:** {mismatch.severity.value.upper()}",
        "",
        f"**Reason:** {mismatch.reason}",
    ]
    if mismatch.critical_missing:
        names = ", ".join(tax.display_name(s) for s in mismatch.critical_missing)
        lines.append(f"**Missing critical skills:** {names}")
    lines.append(
        "\n_The scores below are penalised accordingly — a keyword-dense resume "
        "from a different field should not rank as a fit._"
    )
    return "\n".join(lines)


def _format_ats(result: AnalysisResult) -> str:
    lines = [
        f"## 🎯 ATS SCORE: {result.ats_score:.0f}/100 "
        f"{_score_emoji(result.ats_score)} _{result.verdict}_",
        "```",
    ]
    for name, value in result.ats_breakdown.as_dict().items():
        lines.append(f"{name:<28}{bar(value)} {value:>5.0f}%")
    lines.append("```")
    lines.append(result.ats_explanation)
    return "\n".join(lines)


def _format_skills(result: AnalysisResult) -> str:
    lines = [
        f"## 🧩 SKILL SET MATCH: {result.skill_match_score:.0f}% "
        f"{_score_emoji(result.skill_match_score)}",
        "_Bold = required by the JD. Partial matches count as half credit._",
    ]

    if result.matched_skills:
        lines.append(
            f"\n✅ **Strong matches ({len(result.matched_skills)})**\n"
            f"{_skill_names(result.matched_skills)}"
        )
    else:
        lines.append("\n✅ **Strong matches** — none found.")

    if result.partial_skills:
        lines.append(f"\n🟡 **Partial / related matches ({len(result.partial_skills)})**")
        for match in result.partial_skills[:6]:
            lines.append(
                f"• **{tax.display_name(match.skill)}** ← "
                f"{tax.display_name(match.matched_via)} _({match.relation})_"
            )
        if len(result.partial_skills) > 6:
            lines.append(f"_…and {len(result.partial_skills) - 6} more._")

    if result.missing_skills:
        required_missing = [
            m for m in result.missing_skills if m.importance == SkillImportance.REQUIRED
        ]
        preferred_missing = [
            m for m in result.missing_skills if m.importance == SkillImportance.PREFERRED
        ]
        lines.append(f"\n❌ **Missing skills ({len(result.missing_skills)})**")
        if required_missing:
            lines.append(f"_Required:_ {_skill_names(required_missing, 12)}")
        if preferred_missing:
            lines.append(f"_Preferred:_ {_skill_names(preferred_missing, 10)}")

    return "\n".join(lines)


def _format_missing_areas(areas: list[MissingArea]) -> str:
    if not areas:
        return (
            "## ✅ MISSING / IMPROVEMENT AREAS\n"
            "No significant gaps were found: every required skill is evidenced and "
            "all alignment dimensions scored well."
        )

    lines = ["## ❌ MISSING / IMPROVEMENT AREAS"]
    ordered = sorted(areas, key=lambda a: _PRIORITY_ORDER[a.priority])

    shown = 0
    for priority in (Priority.HIGH, Priority.MEDIUM, Priority.LOW):
        group = [a for a in ordered if a.priority == priority]
        if not group:
            continue
        lines.append(f"\n**{_PRIORITY_LABEL[priority]}**")
        for area in group[:6]:
            label = tax.display_name(area.area) if len(area.area) < 40 else area.area
            category = area.category.replace("_", " ")
            lines.append(f"• **{label}** _({category})_")
            if area.detail:
                lines.append(f"  ↳ {area.detail}")
            shown += 1
        if len(group) > 6:
            lines.append(f"  _…and {len(group) - 6} more._")
        if shown >= 12:
            break

    return "\n".join(lines)


def _format_alignment(result: AnalysisResult) -> str:
    lines = [
        f"## 📈 OVERALL JOB ALIGNMENT: {result.alignment_score:.0f}% "
        f"{_score_emoji(result.alignment_score)}",
        "```",
    ]
    for name, value in result.alignment_breakdown.as_dict().items():
        lines.append(f"{name:<20}{bar(value)} {value:>5.0f}%")
    lines.append("```")
    lines.append(result.alignment_explanation)

    if result.strong_points:
        lines.append("\n**Strong alignment**")
        for point in result.strong_points:
            lines.append(f"• {point}")
    if result.weak_points:
        lines.append("\n**Weak alignment**")
        for point in result.weak_points:
            lines.append(f"• {point}")

    # Derived from the breakdown printed directly above, so this section can
    # never claim a clean bill of health while a dimension scores 30%.
    from app.analysis.analyzer import weak_alignment_dimensions

    weak = weak_alignment_dimensions(result.alignment_breakdown)
    if weak:
        lines.append("\n⚠️ **KEY ALIGNMENT GAPS**")
        for name, score in weak:
            lines.append(f"• {name} alignment is low ({score:.0f}%).")
    return "\n".join(lines)


def _format_courses(result: AnalysisResult) -> str:
    if not result.recommended_courses:
        return (
            "## 📚 RECOMMENDED COURSES\n"
            "No courses are necessary — the resume already evidences every skill "
            "this JD asks for."
        )

    lines = ["## 📚 RECOMMENDED COURSES"]
    for number, course in enumerate(result.recommended_courses, start=1):
        provider = f" — _{course.provider}_" if course.provider else ""
        flag = {"high": "🔴", "medium": "🟠", "low": "🟡"}[course.priority.value]
        # Discord renders [text](url); fall back to plain text if no link.
        title = f"[{course.title}](<{course.url}>)" if course.url else course.title
        lines.append(f"**{number}. {title}**{provider} {flag}")
        lines.append(f"   ↳ **Why:** {course.reason}")
    return "\n".join(lines)


def _format_roadmap(result: AnalysisResult) -> str:
    if not result.learning_roadmap:
        return ""
    lines = ["## 🧠 LEARNING ROADMAP"]
    for item in result.learning_roadmap:
        lines.append(f"**Priority {item.priority} — {tax.display_name(item.skill)}**")
        if item.topics:
            lines.append("Learn: " + ", ".join(item.topics))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Message packing
# --------------------------------------------------------------------------- #
def pack_messages(blocks: list[str], limit: int = _SAFE_LIMIT) -> list[str]:
    """Pack blocks into Discord-sized messages without breaking code fences."""
    messages: list[str] = []
    current = ""

    for block in blocks:
        if not block:
            continue
        pieces = [block] if len(block) <= limit else _split_block(block, limit)
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 2 + len(piece) <= limit:
                current = f"{current}\n\n{piece}"
            else:
                messages.append(current)
                current = piece

    if current:
        messages.append(current)
    return messages


def _split_block(block: str, limit: int) -> list[str]:
    """Split an oversized block on line boundaries, keeping code fences balanced."""
    parts: list[str] = []
    current: list[str] = []
    length = 0
    in_fence = False

    for line in block.split("\n"):
        line_length = len(line) + 1
        would_overflow = length + line_length > limit

        if would_overflow and current:
            chunk = "\n".join(current)
            if in_fence:
                # Close the fence here and reopen it in the next chunk.
                chunk += "\n```"
            parts.append(chunk)
            current = ["```"] if in_fence else []
            length = 4 if in_fence else 0

        if line.strip().startswith("```"):
            in_fence = not in_fence

        current.append(line)
        length += line_length

    if current:
        chunk = "\n".join(current)
        if in_fence:
            chunk += "\n```"
        parts.append(chunk)

    # A single line longer than the limit still has to be cut somewhere.
    final: list[str] = []
    for part in parts:
        while len(part) > limit:
            final.append(part[:limit])
            part = part[limit:]
        if part:
            final.append(part)
    return final


def format_full_report(report: ComparisonReport) -> list[str]:
    """Complete recruiter-facing output: comparison table then each analysis."""
    blocks: list[str] = []
    if len(report.results) > 1:
        blocks.extend(format_comparison(report))
        blocks.append("─" * 40 + "\n## 📑 DETAILED ANALYSIS")

    for index, result in enumerate(report.ranked, start=1):
        blocks.append("─" * 40)
        blocks.extend(format_result(result, index if len(report.results) > 1 else None))

    return pack_messages(blocks)
