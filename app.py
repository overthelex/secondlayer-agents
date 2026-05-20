"""LMAF (Legal Multi-Agent Framework) Gradio chat app.

Chat interface for multi-agent legal consultation pipeline.
Runs on prod (agents.legal.org.ua) and HuggingFace Space.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr


EXAMPLES = [
    "Покупець не оплатив товар на 150 000 грн протягом 6 місяців. Як стягнути пеню, 3% річних та інфляційні?",
    "Працівника звільнено під час воєнного стану без попередження. Чи є підстави для поновлення?",
    "Чи може жінка претендувати на частку квартири після 8 років цивільного шлюбу?",
    "Забудовник затримує введення будинку в експлуатацію на 2 роки. Які компенсації можна вимагати?",
]


def has_api_keys() -> bool:
    return bool(
        os.environ.get("AWS_REGION")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


async def _run_pipeline(question: str):
    """Run the multi-agent pipeline and yield status updates."""
    from lmaf.core.config import Config
    from lmaf.engine import LMAF
    from lmaf.state.research_state import CritiqueStatus

    config = Config.from_env()
    config.max_iterations = 2
    config.critic_every_n = 2
    lmaf = LMAF(question, config)

    yield "Surveyor: аналізую правовий ландшафт..."
    await lmaf.surveyor.run(lmaf.state)
    lmaf._loop.survey_done = True
    if lmaf.state.survey_summary:
        yield f"**Огляд**: {lmaf.state.survey_summary[:300]}"

    yield "Planner: розробляю стратегію дослідження..."
    await lmaf.planner.run(lmaf.state)
    lmaf._loop.plan_done = True
    if lmaf.state.strategy.approach:
        yield f"**Стратегія**: {lmaf.state.strategy.approach[:300]}"

    for iteration in range(1, config.max_iterations + 1):
        lmaf.state.iteration = iteration

        yield f"Ітерація {iteration}: Orchestrator вирішує наступний крок..."
        orch_result = await lmaf.orchestrator.run(lmaf.state)

        if not orch_result.success:
            if lmaf._loop.consecutive_failures >= config.max_consecutive_failures:
                yield "Занадто багато послідовних помилок, зупиняюсь."
                break
            continue

        yield "Researcher: шукаю судову практику та законодавство..."
        agent_result = await lmaf.researcher.run(
            lmaf.state, task_description=orch_result.summary
        )
        if agent_result.success:
            yield f"**Знайдено**: {agent_result.summary[:200]}"

        yield "Reviewer: верифікую докази та посилання..."
        await lmaf.reviewer.run(lmaf.state)

        if iteration % config.critic_every_n == 0:
            yield "Critic: аудит стратегії та повноти аналізу..."
            critic_result = await lmaf.critic.run(lmaf.state)
            lmaf._loop.last_critic_iteration = iteration

            for critique in lmaf.state.active_critiques():
                if critique.type == "strategy":
                    await lmaf.planner.run(
                        lmaf.state, revision_critique=critique.details
                    )
                    critique.status = CritiqueStatus.RESOLVED

            if "можна завершувати: True" in critic_result.summary:
                yield "Critic схвалив -- формую консультацію."
                break

        if not lmaf.state.open_questions() and not lmaf.state.active_critiques():
            yield "Всі питання вирішено -- формую консультацію."
            break

    yield "Formatter: оформлюю фінальну консультацію..."
    await lmaf.formatter.run(lmaf.state)

    yield lmaf.state.answer or "Не вдалося сформувати відповідь."


async def stream_chat(message: str, history: list[dict]):
    """Async streaming chat handler -- yields incremental updates."""
    if not message.strip():
        yield "Будь ласка, опишіть вашу правову ситуацію."
        return

    if not has_api_keys():
        yield (
            "API ключі не налаштовано. Цей інстанс працює в демо-режимі.\n\n"
            "Для реальних консультацій використовуйте "
            "[agents.legal.org.ua](https://agents.legal.org.ua)."
        )
        return

    accumulated = ""
    status_prefixes = (
        "Surveyor", "Planner", "Ітерація", "Researcher",
        "Reviewer", "Critic", "Formatter", "Всі", "Занадто",
    )
    async for update in _run_pipeline(message):
        if update.startswith("**") or update.startswith("#"):
            accumulated += f"\n\n{update}"
        elif update.startswith(status_prefixes):
            accumulated += f"\n\n__{update}__"
        else:
            accumulated = update
        yield accumulated


_arch_raw = (Path(__file__).parent / "architecture.html").read_text(encoding="utf-8")
import html as _html_mod
_arch_escaped = _html_mod.escape(_arch_raw, quote=True)
ARCHITECTURE_HTML = (
    f'<iframe srcdoc="{_arch_escaped}" '
    'style="width:100%;height:700px;border:none;border-radius:12px;" '
    'sandbox="allow-scripts allow-same-origin"></iframe>'
)


DATASETS_MD = """
## Пов'язані датасети на HuggingFace

| Датасет | Розмір | Опис |
|---------|--------|------|
| [ua-case-outcome-6m](https://huggingface.co/datasets/overthelex/ua-case-outcome-6m) | 6.7M | Повний датасет рішень суду з темпоральними спліттами |
| [ukrainian-court-decisions](https://huggingface.co/datasets/overthelex/ukrainian-court-decisions) | 428K | Збалансований бенчмарк для LEXTREME |
| [ua-court-citation-graph](https://huggingface.co/datasets/overthelex/ua-court-citation-graph) | 2.3M | Граф ко-цитування з 99.5М рішень |
| [ua-statute-retrieval](https://huggingface.co/datasets/overthelex/ua-statute-retrieval) | 396M citations | Бенчмарк пошуку законодавства |
| [ua-temporal-drift](https://huggingface.co/datasets/overthelex/ua-temporal-drift) | 428K | Дані про темпоральний дрифт |

## Пов'язані статті

- [Temporal Decay of Co-Citation Predictability](https://arxiv.org/abs/2605.17639) (arXiv, 2025)
- [A Citation Graph from 100M Court Decisions](https://arxiv.org/abs/2605.15362) (arXiv, 2025)
- [Tokenizer Fertility on Ukrainian Legal Text](https://arxiv.org/abs/2605.14890) (arXiv, 2025)
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LMAF") as app:
        gr.Markdown(
            "# LMAF -- Legal Multi-Agent Framework\n"
            "### Мульти-агентна система для складних правових консультацій\n\n"
            "Дев'ять спеціалізованих LLM-агентів аналізують правові питання, "
            "шукають судову практику у 100М+ рішень ЄДРСР та формують структуровану консультацію.\n\n"
            "*[GitHub](https://github.com/overthelex/secondlayer-agents) | "
            "[SecondLayer](https://legal.org.ua) | "
            "[Датасети](https://huggingface.co/overthelex)*"
        )

        with gr.Tabs():
            with gr.Tab("Чат"):
                chatbot = gr.ChatInterface(
                    fn=stream_chat,
                    examples=EXAMPLES,
                    title=None,
                    chatbot=gr.Chatbot(
                        height=600,
                        placeholder="Опишіть правову ситуацію -- 9 агентів проаналізують та підготують консультацію",
                    ),
                    textbox=gr.Textbox(
                        placeholder="Опишіть вашу правову ситуацію...",
                        scale=7,
                    ),
                )

            with gr.Tab("Архітектура"):
                gr.HTML(ARCHITECTURE_HTML)

            with gr.Tab("Датасети"):
                gr.Markdown(DATASETS_MD)

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
