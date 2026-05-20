"""Render ConsultationState to Markdown files for workspace snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..state.research_state import ConsultationState


def render_state_md(state: ConsultationState) -> str:
    """Render the full consultation state as Markdown."""
    parts = []
    parts.append(f"# {state.title}")
    parts.append(f"\n## Запит клієнта\n{state.client_question}")

    if state.jurisdiction:
        parts.append(f"**Юрисдикція**: {state.jurisdiction}")

    if state.survey_summary:
        parts.append(f"\n## Огляд\n{state.survey_summary}")

    if state.strategy.approach:
        parts.append(f"\n## Стратегія\n{state.strategy.approach}")

    if state.hypotheses:
        parts.append("\n## Правові позиції")
        for h in state.hypotheses:
            parts.append(f"- **{h.id}** ({h.status}): {h.statement}")

    if state.evidence:
        parts.append("\n## Докази")
        for ev in state.evidence:
            prefix = "~~" if ev.refuted else ""
            suffix = "~~" if ev.refuted else ""
            parts.append(f"- {prefix}**{ev.id}** [{ev.type}] {ev.citation}: {ev.summary}{suffix}")

    if state.questions:
        parts.append("\n## Дослідницькі питання")
        for q in state.questions:
            parts.append(f"- **{q.id}** ({q.status}): {q.question}")
            if q.answer:
                parts.append(f"  > {q.answer[:200]}")

    if state.critiques:
        parts.append("\n## Зауваження")
        for c in state.critiques:
            parts.append(f"- [{c.status}] **{c.id}** ({c.severity}): {c.summary}")

    parts.append(f"\n---\nIteration: {state.iteration}")

    return "\n".join(parts)
