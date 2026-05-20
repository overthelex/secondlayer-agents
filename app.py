"""LegalIntern Gradio chat app.

Chat interface for multi-agent legal consultation pipeline.
On prod (agents.legal.org.ua): runs the real pipeline with LLM + SecondLayer API.
On HF Space: proxies to prod via gr.load().
"""

from __future__ import annotations

import asyncio
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
    from legal_intern.core.config import Config
    from legal_intern.engine import LegalIntern

    config = Config.from_env()
    intern = LegalIntern(question, config)

    yield "Surveyor: аналізую правовий ландшафт..."
    result = await intern.surveyor.run(intern.state)
    intern._loop.survey_done = True
    if intern.state.survey_summary:
        yield f"**Огляд**: {intern.state.survey_summary[:300]}"

    yield "Planner: розробляю стратегію дослідження..."
    result = await intern.planner.run(intern.state)
    intern._loop.plan_done = True
    if intern.state.strategy.approach:
        yield f"**Стратегія**: {intern.state.strategy.approach[:300]}"

    for iteration in range(1, config.max_iterations + 1):
        intern.state.iteration = iteration

        yield f"Ітерація {iteration}: Orchestrator вирішує наступний крок..."
        orch_result = await intern.orchestrator.run(intern.state)

        if not orch_result.success:
            if intern._loop.consecutive_failures >= config.max_consecutive_failures:
                yield "Занадто багато послідовних помилок, зупиняюсь."
                break
            continue

        yield f"Researcher: шукаю судову практику та законодавство..."
        agent_result = await intern.researcher.run(
            intern.state, task_description=orch_result.summary
        )
        if agent_result.success:
            yield f"**Знайдено**: {agent_result.summary[:200]}"

        yield "Reviewer: верифікую докази та посилання..."
        review_result = await intern.reviewer.run(intern.state)

        if iteration % config.critic_every_n == 0:
            yield "Critic: аудит стратегії та повноти аналізу..."
            critic_result = await intern.critic.run(intern.state)
            intern._loop.last_critic_iteration = iteration

            for critique in intern.state.active_critiques():
                if critique.type == "strategy":
                    await intern.planner.run(
                        intern.state, revision_critique=critique.details
                    )
                    from legal_intern.state.research_state import CritiqueStatus
                    critique.status = CritiqueStatus.RESOLVED

            if "можна завершувати: True" in critic_result.summary:
                yield "Critic схвалив -- формую консультацію."
                break

        if not intern.state.open_questions() and not intern.state.active_critiques():
            yield "Всі питання вирішено -- формую консультацію."
            break

    yield "Formatter: оформлюю фінальну консультацію..."
    await intern.formatter.run(intern.state)

    yield intern.state.answer or "Не вдалося сформувати відповідь."


def chat_fn(message: str, history: list[dict]) -> str:
    """Synchronous wrapper that runs the async pipeline."""
    if not message.strip():
        return "Будь ласка, опишіть вашу правову ситуацію."

    if not has_api_keys():
        return (
            "API ключі не налаштовано. Цей інстанс працює в демо-режимі.\n\n"
            "Для реальних консультацій використовуйте "
            "[agents.legal.org.ua](https://agents.legal.org.ua)."
        )

    final = ""
    for update in _run_sync(message):
        final = update
    return final


def _run_sync(question: str):
    """Run async generator synchronously, collecting results."""
    loop = asyncio.new_event_loop()
    gen = _run_pipeline(question)
    try:
        while True:
            result = loop.run_until_complete(gen.__anext__())
            yield result
    except StopAsyncIteration:
        pass
    finally:
        loop.close()


def stream_chat(message: str, history: list[dict]):
    """Streaming chat handler -- yields incremental updates."""
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
    for update in _run_sync(message):
        if update.startswith("**") or update.startswith("#"):
            accumulated += f"\n\n{update}"
        elif not update.startswith("Surveyor") and not update.startswith("Planner") \
                and not update.startswith("Ітерація") and not update.startswith("Researcher") \
                and not update.startswith("Reviewer") and not update.startswith("Critic") \
                and not update.startswith("Formatter") and not update.startswith("Всі") \
                and not update.startswith("Занадто"):
            accumulated = update
        else:
            accumulated += f"\n\n__{update}__"
        yield accumulated


ARCHITECTURE_MD = """
## Архітектура LegalIntern

Дев'ять спеціалізованих LLM-агентів працюють у циклі. Кожен агент починає з чистого контексту.
Весь стан зберігається у структурованому об'єкті `ConsultationState`.

```
Запит клієнта
     │
     ▼
┌─────────────┐
│  Surveyor   │  Огляд правового ландшафту
└──────┬──────┘
       ▼
┌─────────────┐
│   Planner   │  Стратегія дослідження
└──────┬──────┘
       ▼
┌─────────────┐     ┌────────────┐
│Orchestrator │────►│ Researcher │  Судова практика, НПА
│ (dispatch)  │────►│  Analyst   │  Обчислення, строки
└──────┬──────┘     └─────┬──────┘
       │                  │
       ▼                  ▼
┌─────────────┐   ┌────────────┐
│  Reviewer   │   │   Critic   │  Аудит стратегії
│(верифікація)│   │ (periodic) │
└──────┬──────┘   └─────┬──────┘
       │                │
       ▼                ▼
┌─────────────┐  ┌──────────────┐
│ Adjudicator │  │   Planner    │
│  (арбітраж) │  │  (revision)  │
└──────┬──────┘  └──────────────┘
       ▼
┌─────────────┐
│  Formatter  │  Фінальна консультація
└─────────────┘
```

| Агент | Роль | Аналог в secondlayer-core |
|-------|------|--------------------------|
| **Surveyor** | Карта правового ландшафту | IntentClassifier + QueryPlanner |
| **Planner** | Стратегія дослідження | Генерація ExecutionPlan |
| **Orchestrator** | Координація та гіпотези | Агентний цикл ChatService |
| **Researcher** | Пошук практики та НПА | Виклики інструментів |
| **Analyst** | Обчислення строків, сум | Калькуляційні інструменти |
| **Reviewer** | Верифікація доказів | CitationValidator + HallucinationGuard |
| **Critic** | Аудит стратегії | *Нова можливість* |
| **Adjudicator** | Арбітраж розбіжностей | *Нова можливість* |
| **Formatter** | Оформлення консультації | Синтез відповіді |
"""


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
    with gr.Blocks(title="LegalIntern") as app:
        gr.Markdown(
            "# LegalIntern\n"
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
                gr.Markdown(ARCHITECTURE_MD)

            with gr.Tab("Датасети"):
                gr.Markdown(DATASETS_MD)

    return app


if __name__ == "__main__":
    is_hf_space = bool(os.environ.get("SPACE_ID"))

    if is_hf_space and not has_api_keys():
        demo = gr.load("https://agents.legal.org.ua")
    else:
        demo = build_app()

    demo.launch()
