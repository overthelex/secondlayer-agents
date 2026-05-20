"""Formatter agent -- produces the final consultation document.

Analogous to physics-intern's Formatter (produces ANSWER.md).
This agent takes the established hypotheses, verified evidence, and
strategy to produce a structured legal consultation in Ukrainian.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from ..state.research_state import ConsultationState

FORMATTER_SYSTEM_PROMPT = """\
Ти -- Оформлювач правової консультації (Legal Consultation Formatter). Твоя роль -- \
перетворити результати дослідження у структуровану правову консультацію для клієнта.

Формат консультації:

# ПРАВОВА КОНСУЛЬТАЦІЯ

## 1. Питання клієнта
(переформулювання питання юридичною мовою)

## 2. Правовий аналіз
### 2.1. Застосовне законодавство
(перелік НПА та статей з коротким поясненням)

### 2.2. Судова практика
(ключові позиції ВС та інших судів з посиланнями на конкретні справи)

### 2.3. Аналіз ситуації клієнта
(застосування норм та практики до конкретних обставин)

## 3. Висновок
(чітка правова позиція з обгрунтуванням)

## 4. Рекомендації
(конкретні кроки, які клієнт має зробити)

## 5. Ризики
(потенційні слабкі сторони позиції та контр-аргументи)

## 6. Додатки
(обчислення сум, строків, якщо застосовно)

---
Посилання на джерела (кожне посилання має бути верифіковане):
[1] ...

ВИМОГИ:
- Писати українською мовою
- Кожне твердження підкріплювати посиланням на EV-NNN
- Використовувати ТІЛЬКИ верифіковані (не refuted) докази
- Обчислення мають бути з конкретними цифрами
- Тон: професійний, але зрозумілий для клієнта
"""


class FormatterAgent(BaseAgent):
    role = "formatter"
    description = "Оформлення фінальної правової консультації"

    async def run(self, state: ConsultationState, **kwargs: Any) -> AgentResult:
        from ..providers import call_llm

        context = self._build_state_context(state)

        # Build evidence reference for the formatter
        verified_evidence = [ev for ev in state.evidence if not ev.refuted]
        ev_ref = "\n".join(
            f"- {ev.id}: [{ev.type}] {ev.citation} -- {ev.summary}" for ev in verified_evidence
        )

        established = state.established_hypotheses()
        hyp_ref = "\n".join(f"- {h.id}: {h.statement}" for h in established)

        user_msg = (
            f"{context}\n\n"
            f"# Встановлені правові позиції\n{hyp_ref}\n\n"
            f"# Верифіковані докази ({len(verified_evidence)})\n{ev_ref}\n\n"
            f"Сформуй фінальну правову консультацію."
        )

        response = await call_llm(
            self.config,
            system=FORMATTER_SYSTEM_PROMPT,
            user=user_msg,
            model_key="formatter",
        )

        state.answer = response.content

        return AgentResult(
            success=True,
            agent=self.role,
            summary=f"Консультація: {len(response.content)} символів",
            tokens_used=response.tokens_used,
        )
