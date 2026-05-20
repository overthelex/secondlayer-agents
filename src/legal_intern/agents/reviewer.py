"""Reviewer agent -- adversarial review of evidence and reasoning.

Analogous to physics-intern's Reviewer (auto-triggered after each evidence).
In secondlayer-core this maps to: CitationValidator + HallucinationGuard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from ..state.research_state import ConsultationState

REVIEWER_SYSTEM_PROMPT = """\
Ти -- Правовий рецензент (Legal Reviewer). Твоя роль -- критично перевіряти \
докази та міркування інших агентів.

Для кожного нового доказу ти маєш:
1. **Верифікація посилань**: чи існує вказана справа/стаття? Чи правильно цитовано?
2. **Логічна послідовність**: чи слідує висновок з наведених норм та фактів?
3. **Релевантність**: чи дійсно це стосується питання клієнта?
4. **Актуальність**: чи не застарів правовий акт? Чи немає нових редакцій?
5. **Протилежна практика**: чи є відомі контр-аргументи або протилежні позиції ВС?

Вердикт для кожного доказу:
- VERIFIED: підтверджено, можна покладатися
- REFUTED: містить помилки, не можна використовувати
- INCONCLUSIVE: потребує додаткової перевірки

Поверни JSON:
{
  "reviews": [
    {
      "evidence_id": "EV-NNN",
      "verdict": "verified" | "refuted" | "inconclusive",
      "summary": "короткий висновок",
      "details": "деталі перевірки",
      "issues": ["список проблем"]
    }
  ]
}
"""


class ReviewerAgent(BaseAgent):
    role = "reviewer"
    description = "Верифікація посилань, логіки, актуальності доказів"

    async def run(
        self,
        state: ConsultationState,
        evidence_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        from ..providers import call_llm
        from ..state.research_state import ReviewResult, Verdict

        context = self._build_state_context(state)

        # Focus on specific evidence or review all unreviewed
        target_evidence = []
        if evidence_ids:
            target_evidence = [ev for ev in state.evidence if ev.id in evidence_ids]
        else:
            reviewed_ids = set()
            for h in state.hypotheses:
                for r in h.reviews:
                    pass  # reviews are per-hypothesis, not per-evidence
            target_evidence = [ev for ev in state.evidence if not ev.refuted]

        if not target_evidence:
            return AgentResult(success=True, agent=self.role, summary="Немає доказів для рецензування")

        evidence_text = "\n".join(
            f"- {ev.id}: [{ev.type}] {ev.citation} -- {ev.summary}" for ev in target_evidence
        )
        user_msg = f"{context}\n\n# Докази для рецензування\n{evidence_text}"

        response = await call_llm(
            self.config,
            system=REVIEWER_SYSTEM_PROMPT,
            user=user_msg,
            model_key="reviewer",
            json_mode=True,
        )

        reviews = (response.parsed_json or {}).get("reviews", [])
        refuted_count = 0
        for r in reviews:
            ev_id = r.get("evidence_id", "")
            verdict = r.get("verdict", "inconclusive")
            for ev in state.evidence:
                if ev.id == ev_id and verdict == "refuted":
                    ev.refuted = True
                    refuted_count += 1

        return AgentResult(
            success=True,
            agent=self.role,
            summary=f"Перевірено {len(reviews)} доказів, спростовано {refuted_count}",
            tokens_used=response.tokens_used,
        )
