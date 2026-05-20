"""Adjudicator agent -- resolves disagreements between agents.

Analogous to physics-intern's Adjudicator.
Invoked when a critique challenges an established hypothesis or
when reviewer and researcher disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from ..state.research_state import ConsultationState

ADJUDICATOR_SYSTEM_PROMPT = """\
Ти -- Арбітр правового дослідження (Legal Adjudicator). Тебе викликають коли \
є розбіжність між агентами або коли зауваження оскаржує встановлену позицію.

Ти отримуєш:
- Оскаржувану гіпотезу з її доказами
- Зауваження, що її оскаржує
- Контекст дослідження

Ти маєш:
1. Об'єктивно оцінити обидві сторони
2. Вирішити, чи гіпотеза залишається "established" чи повертається у "working"
3. Якщо зауваження обгрунтоване -- вказати, які додаткові дослідження потрібні

Поверни JSON:
{
  "decision": "uphold" | "reopen" | "refute",
  "reasoning": "обгрунтування рішення",
  "additional_research": ["список додаткових питань"] | null
}
"""


class AdjudicatorAgent(BaseAgent):
    role = "adjudicator"
    description = "Арбітраж розбіжностей між агентами"

    async def run(
        self,
        state: ConsultationState,
        hypothesis_id: str = "",
        critique_id: str = "",
        **kwargs: Any,
    ) -> AgentResult:
        from ..providers import call_llm
        from ..state.research_state import CritiqueStatus, HypothesisStatus

        context = self._build_state_context(state)

        # Find the disputed hypothesis and critique
        hyp = next((h for h in state.hypotheses if h.id == hypothesis_id), None)
        crit = next((c for c in state.critiques if c.id == critique_id), None)

        dispute_text = ""
        if hyp:
            supporting = [
                ev for ev in state.evidence if ev.id in hyp.supporting_evidence and not ev.refuted
            ]
            dispute_text += f"\n# Оскаржувана позиція: {hyp.statement}\n"
            dispute_text += f"Підтверджуючі докази: {', '.join(ev.citation for ev in supporting)}\n"
        if crit:
            dispute_text += f"\n# Зауваження: {crit.summary}\n{crit.details}"

        user_msg = f"{context}\n{dispute_text}"

        response = await call_llm(
            self.config,
            system=ADJUDICATOR_SYSTEM_PROMPT,
            user=user_msg,
            model_key="adjudicator",
            json_mode=True,
        )

        result = response.parsed_json or {}
        decision = result.get("decision", "uphold")

        if hyp:
            if decision == "refute":
                hyp.status = HypothesisStatus.REFUTED
            elif decision == "reopen":
                hyp.status = HypothesisStatus.WORKING

        if crit:
            crit.status = CritiqueStatus.RESOLVED

        for q in result.get("additional_research", []) or []:
            state.add_question(q, iteration=state.iteration)

        return AgentResult(
            success=True,
            agent=self.role,
            summary=f"Рішення: {decision} для {hypothesis_id}",
            tokens_used=response.tokens_used,
        )
