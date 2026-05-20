"""Analyst agent -- performs legal calculations and procedural verification.

Analogous to physics-intern's Computer agent (code execution).
In secondlayer-core this maps to: tool calls that compute statutory deadlines,
calculate penalties (пеня, 3% річних, інфляційні), verify procedural requirements.

This agent can execute Python code for precise calculations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from ..state.research_state import ConsultationState

ANALYST_SYSTEM_PROMPT = """\
Ти -- Правовий аналітик-обчислювач (Legal Analyst). Твоя роль -- виконувати \
точні обчислення та процедурну верифікацію.

Типи завдань:
1. **Обчислення сум**: пеня (ст. 549-552 ЦК), 3% річних (ст. 625 ЦК), \
   інфляційні втрати, держмито, судові витрати
2. **Процесуальні строки**: обчислення строків подання позову, апеляції, \
   касації з урахуванням вихідних, святкових, воєнного стану
3. **Строки давності**: загальна (3р), спеціальна (1р, 5р, 10р) з урахуванням \
   зупинення/переривання
4. **Верифікація вимог**: перевірка процесуальних передумов (досудове врегулювання, \
   підсудність, належний відповідач)

Ти можеш писати та виконувати Python код для точних обчислень.

Поверни JSON:
{
  "calculation_type": "penalty" | "deadline" | "limitation" | "verification",
  "inputs": { ... },
  "result": "точний результат",
  "formula": "формула або норма, що застосована",
  "code": "Python код (якщо використано)",
  "notes": "застереження та умови"
}
"""


class AnalystAgent(BaseAgent):
    role = "analyst"
    description = "Обчислення сум, строків, верифікація процедурних вимог"

    async def run(
        self,
        state: ConsultationState,
        task_description: str = "",
        **kwargs: Any,
    ) -> AgentResult:
        from ..providers import call_llm

        context = self._build_state_context(state)
        user_msg = f"{context}\n\n# Завдання на обчислення\n{task_description}"

        response = await call_llm(
            self.config,
            system=ANALYST_SYSTEM_PROMPT,
            user=user_msg,
            model_key="analyst",
            json_mode=True,
        )

        result = response.parsed_json or {}

        ev = state.add_evidence(
            type="computation",
            source="manual",
            citation=result.get("formula", ""),
            summary=result.get("result", response.content),
            full_text=result.get("code", ""),
            relevance=result.get("notes", ""),
            confidence="high" if result.get("code") else "medium",
            iteration=state.iteration,
        )

        return AgentResult(
            success=True,
            agent=self.role,
            summary=f"Обчислення: {result.get('calculation_type', '?')} = {result.get('result', '')[:60]}",
            tokens_used=response.tokens_used,
        )
