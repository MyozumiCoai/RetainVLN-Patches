from __future__ import annotations

from typing import List, Optional

CHARS_PER_TOKEN = 3.5


def truncate(text: str, budget_tokens: int) -> str:
    max_chars = int(budget_tokens * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2):]
    return f"{head}\n...\n{tail}"


def _bullet(items: List[str], label: str, max_items: int) -> str:
    if not items:
        return ""
    kept = items[-max_items:]
    return f"- {label}: {' | '.join(kept)}"


def build_memory_text(
    progress: str,
    salient: List[str],
    outcome: str,
    *,
    max_salient: int = 3,
) -> str:
    lines = ["Memory:"]
    if progress:
        lines.append(f"- Progress: {progress}")
    lines.append(_bullet(salient, "Events", max_salient))
    if outcome:
        lines.append(f"- Outcome: {outcome}")
    return "\n".join(l for l in lines if l)


def build_notebook_text(
    searched: List[str],
    candidates: List[str],
    rejected: List[str],
    *,
    max_each: int = 3,
) -> str:
    lines = ["Notebook:"]
    lines.append(_bullet(searched, "Searched", max_each))
    lines.append(_bullet(candidates, "Candidates", max_each))
    lines.append(_bullet(rejected, "Rejected", max_each))
    return "\n".join(l for l in lines if l)


def build_memory_context(
    episode: dict,
    notebook: dict,
    *,
    budget_tokens: int = 400,
) -> str:
    mem = build_memory_text(
        episode.get("progress", ""),
        episode.get("salient", []),
        episode.get("outcome", ""),
    )
    nb = build_notebook_text(
        notebook.get("searched_regions", []),
        notebook.get("candidate_locations", []),
        notebook.get("rejected_hypotheses", []),
    )
    combined = "\n".join(x for x in (mem, nb) if x and x not in ("Memory:", "Notebook:"))
    return truncate(combined, budget_tokens)
