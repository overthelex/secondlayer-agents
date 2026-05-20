"""Gradio app for the LegalIntern HuggingFace Space.

Interactive demo of the multi-agent legal consultation pipeline.
Shows the agent flow, state evolution, and final consultation.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import gradio as gr

from src.legal_intern.state.research_state import (
    ConsultationState,
    LegalEvidence,
    LegalHypothesis,
    LegalStrategy,
)
from src.legal_intern.rendering import render_state_md


EXAMPLE_QUESTIONS = [
    "Покупець не оплатив товар на 150 000 грн протягом 6 місяців. Як стягнути пеню, 3% річних та інфляційні?",
    "Працівника звільнено під час воєнного стану без попередження. Чи є підстави для поновлення?",
    "Чи може жінка претендувати на частку квартири після 8 років цивільного шлюбу?",
    "Забудовник затримує введення будинку в експлуатацію на 2 роки. Які компенсації можна вимагати?",
]

ARCHITECTURE_MD = """
## Архітектура LegalIntern

Дев'ять спеціалізованих LLM-агентів працюють у циклі. Кожен агент починає з чистого контексту
(без історії розмови). Весь стан зберігається у структурованому об'єкті `ConsultationState`.

### Pipeline

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

### Агенти

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

### Ключові принципи

1. **Жоден агент не несе історію** -- кожен виклик починає з чистого контексту
2. **Структурований стан** -- гіпотези, докази, зауваження зі зв'язками між ними
3. **Git-версіонування** -- кожна ітерація = коміт, повна відтворюваність
4. **SecondLayer MCP bridge** -- доступ до 100М+ рішень суду, законодавства, позицій ВС

### Порівняння з PhysicsIntern

| Аспект | PhysicsIntern | LegalIntern |
|--------|--------------|-------------|
| Домен | Теоретична фізика | Українське право |
| Researcher | Аналітичні міркування | Пошук у ЄДРСР + НПА |
| Computer | Виконання Python коду | Обчислення пені, строків |
| Верифікація | Formal evaluation | Перевірка посилань |
| Стан | ResearchState (hypotheses) | ConsultationState (правові позиції) |
| Вихід | ANSWER.md | CONSULTATION.md |
"""

DEMO_STATE = ConsultationState(
    client_question="Покупець не оплатив товар на 150 000 грн протягом 6 місяців. Як стягнути пеню?",
    jurisdiction="civil",
    title="Стягнення пені за прострочення оплати",
    survey_summary="Питання стосується цивільно-правової відповідальності за порушення грошового зобов'язання...",
    strategy=LegalStrategy(
        approach="Стягнення на підставі ст. 549-552 ЦК (пеня), ст. 625 ЦК (3% річних + інфляційні)",
        legal_domains=["цивільне право", "зобов'язальне право"],
        key_questions=[
            "Чи передбачена пеня договором?",
            "Який розмір 3% річних та інфляційних?",
            "Яка позиція ВС щодо одночасного стягнення?",
        ],
        relevant_legislation=["ст. 549-552 ЦК України", "ст. 625 ЦК України", "ст. 3 ЗУ 'Про відповідальність за несвоєчасне виконання грошових зобов'язань'"],
    ),
    iteration=5,
)
DEMO_STATE._hyp_counter = 2
DEMO_STATE.hypotheses = [
    LegalHypothesis(
        id="H-001",
        statement="Продавець має право на пеню за ст. 549 ЦК якщо це передбачено договором",
        status="established",
        supporting_evidence=["EV-001", "EV-003"],
    ),
    LegalHypothesis(
        id="H-002",
        statement="3% річних та інфляційні нараховуються незалежно від пені (ст. 625 ЦК)",
        status="established",
        supporting_evidence=["EV-002", "EV-004"],
    ),
]
DEMO_STATE._ev_counter = 4
DEMO_STATE.evidence = [
    LegalEvidence(id="EV-001", type="legislation", source="rada", citation="ст. 549 ЦК України", summary="Неустойкою (штрафом, пенею) є грошова сума, яку боржник повинен передати кредиторові у разі порушення зобов'язання", confidence="high"),
    LegalEvidence(id="EV-002", type="legislation", source="rada", citation="ст. 625 ЦК України", summary="Боржник зобов'язаний сплатити 3% річних та відшкодувати інфляційні втрати", confidence="high"),
    LegalEvidence(id="EV-003", type="case_law", source="edrsr", citation="Справа №757/12345/22", summary="ВС: пеня нараховується з дня, наступного за останнім днем строку оплати", confidence="high"),
    LegalEvidence(id="EV-004", type="case_law", source="edrsr", citation="ВП ВС, справа №910/5678/21", summary="Одночасне стягнення пені та 3% річних є правомірним, оскільки вони мають різну правову природу", confidence="high"),
]


def show_architecture():
    return ARCHITECTURE_MD


def show_demo_state():
    return render_state_md(DEMO_STATE)


def show_demo_json():
    return json.dumps(DEMO_STATE.to_dict(), ensure_ascii=False, indent=2)


def run_demo_consultation(question: str, progress=gr.Progress()):
    """Simulate a consultation run (demo mode without real LLM calls)."""
    if not question.strip():
        return "Будь ласка, введіть правове питання.", "", ""

    progress(0.1, desc="Surveyor: огляд правового ландшафту...")
    progress(0.25, desc="Planner: розробка стратегії...")
    progress(0.4, desc="Researcher: пошук судової практики...")
    progress(0.55, desc="Analyst: обчислення сум...")
    progress(0.7, desc="Reviewer: верифікація доказів...")
    progress(0.85, desc="Critic: аудит стратегії...")
    progress(0.95, desc="Formatter: оформлення консультації...")

    demo_answer = f"""# ПРАВОВА КОНСУЛЬТАЦІЯ

## 1. Питання клієнта
{question}

## 2. Правовий аналіз

> Це демо-режим. У повній версії LegalIntern виконує реальний пошук
> по 100М+ рішень суду в ЄДРСР та аналізує чинне законодавство.

### 2.1. Застосовне законодавство
- Цивільний кодекс України (ст. 549-552, 625)
- Закон України "Про відповідальність за несвоєчасне виконання грошових зобов'язань"

### 2.2. Судова практика
- Позиція Верховного Суду щодо одночасного стягнення пені та 3% річних

## 3. Висновок
Для отримання повної консультації з реальними посиланнями на судову практику
та обчисленнями, запустіть LegalIntern з API ключами SecondLayer та Anthropic.

## 4. Рекомендації
1. Встановіть `ANTHROPIC_API_KEY` та `SECONDLAYER_API_KEY`
2. Запустіть: `legal-intern "{question[:50]}..."`

---
*Demo mode -- real consultations require API access to SecondLayer (legal.org.ua)*
"""

    state_md = f"""# Стан консультації (демо)
- Питання: {question[:80]}...
- Ітерацій: 5 (демо)
- Гіпотез: 2 (встановлено: 2)
- Доказів: 4 (спростовано: 0)
"""

    return demo_answer, state_md, "Demo completed"


# Build Gradio UI
with gr.Blocks(
    title="LegalIntern -- Multi-Agent Legal Consultation",
    theme=gr.themes.Soft(),
    css="""
    .main-header { text-align: center; margin-bottom: 1rem; }
    .agent-flow { font-family: monospace; }
    """,
) as demo:
    gr.Markdown(
        """
        # ⚖️ LegalIntern
        ### Мульти-агентна система для складних правових консультацій

        Дев'ять спеціалізованих LLM-агентів працюють разом для аналізу правових питань,
        пошуку судової практики у 100М+ рішень ЄДРСР, та формування структурованої консультації.

        *Натхнення: [PhysicsIntern](https://huggingface.co/spaces/huggingface/physics-intern) |
        Дані: [SecondLayer](https://legal.org.ua) |
        Код: [GitHub](https://github.com/overthelex/secondlayer-agents)*
        """,
        elem_classes="main-header",
    )

    with gr.Tabs():
        with gr.Tab("Консультація (демо)"):
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Правове питання",
                        placeholder="Опишіть вашу правову ситуацію...",
                        lines=4,
                    )
                    gr.Examples(
                        examples=[[q] for q in EXAMPLE_QUESTIONS],
                        inputs=question_input,
                    )
                    run_btn = gr.Button("Запустити консультацію", variant="primary")

                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="Статус", interactive=False)

            with gr.Row():
                with gr.Column():
                    consultation_output = gr.Markdown(label="Консультація")
                with gr.Column():
                    state_output = gr.Markdown(label="Стан дослідження")

            run_btn.click(
                run_demo_consultation,
                inputs=[question_input],
                outputs=[consultation_output, state_output, status_output],
            )

        with gr.Tab("Архітектура"):
            gr.Markdown(ARCHITECTURE_MD)

        with gr.Tab("ConsultationState (приклад)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Rendered Markdown")
                    gr.Markdown(render_state_md(DEMO_STATE))
                with gr.Column():
                    gr.Markdown("### Raw JSON")
                    gr.Code(
                        json.dumps(DEMO_STATE.to_dict(), ensure_ascii=False, indent=2),
                        language="json",
                    )

        with gr.Tab("Порівняння з PhysicsIntern"):
            gr.Markdown("""
## PhysicsIntern vs LegalIntern

Обидві системи використовують однаковий патерн: 9 спеціалізованих агентів,
структурований стан, git-версіонування, відсутність історії у агентів.

| Компонент | PhysicsIntern | LegalIntern |
|-----------|--------------|-------------|
| **Домен** | Теоретична фізика, математика | Українське цивільне/господарське право |
| **Surveyor** | Огляд наукового ландшафту | Огляд правового ландшафту |
| **Researcher** | Аналітичні деривації | Пошук у ЄДРСР (100М+ рішень) |
| **Computer** | Виконання Python коду | Обчислення пені, строків, інфляційних |
| **Reviewer** | VERIFIED/REFUTED/INCONCLUSIVE | Верифікація посилань + галюцінацій |
| **Critic** | Стратегія + когерентність | Повнота аналізу + контр-аргументи |
| **State** | `ResearchState` (гіпотези, докази) | `ConsultationState` (правові позиції, докази) |
| **Tools** | Python sandbox | SecondLayer MCP API |
| **Output** | `ANSWER.md` | `CONSULTATION.md` |
| **Benchmark** | CritPT (фізика) | UA Court Decisions (6.7M) |
| **LLM** | Multi-provider (Anthropic, OpenAI, Gemini, HF) | Anthropic + OpenAI |

### Що LegalIntern додає

1. **SecondLayer MCP Bridge** -- прямий доступ до ЄДРСР, zakon.rada.gov.ua, ЄСПЛ
2. **Юридична верифікація** -- перевірка існування справ, актуальності НПА
3. **Обчислення** -- пеня (ст. 549-552 ЦК), 3% річних (ст. 625), інфляційні, строки
4. **Українська мова** -- всі промпти та вихід українською

### Що SecondLayer вже має (secondlayer-core)

| Компонент | Опис |
|-----------|------|
| IntentClassifier | Класифікація запиту (LLM + regex fallback) |
| QueryPlanner | Маршрутизація до доменів (court, npa, echr) |
| ChatService | Агентний цикл з tool_use |
| CitationValidator | Перевірка посилань на рішення суду |
| HallucinationGuard | Виявлення фабрикованих номерів справ |
| ShepardizationService | Перевірка актуальності правових позицій |
| EvidenceExtractor | Витяг рішень, цитат, документів |

LegalIntern переосмислює цей pipeline як мульти-агентну систему з
експліцитним станом, критичним оглядом та арбітражем.
            """)

        with gr.Tab("Датасети"):
            gr.Markdown("""
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
            """)

if __name__ == "__main__":
    demo.launch()
