---
title: LMAF
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: true
license: apache-2.0
tags:
  - legal-nlp
  - multi-agent
  - ukraine
  - court-decisions
  - legal-consultation
---

# LMAF -- Legal Multi-Agent Framework

Nine specialised LLM agents analyse Ukrainian legal questions, search 100M+ court decisions in the [EDRSR](https://reyestr.court.gov.ua/) via [SecondLayer](https://legal.org.ua), and produce structured legal consultations with citations to legislation and case law.

**[Live demo](https://agents.legal.org.ua)** &nbsp;|&nbsp; **[HuggingFace Space](https://huggingface.co/spaces/overthelex/lmaf)** &nbsp;|&nbsp; **[SecondLayer Platform](https://legal.org.ua)**

## How It Works

A client question goes through four phases. Every agent starts from a fresh context (no conversation history) -- all state lives in a structured `ConsultationState` object, and every iteration is git-committed for reproducibility.

```
Phase 1        Phase 2                Phase 3                         Phase 4
                              RESEARCH LOOP (up to 15 iterations)
                         ┌─────────────────────────────────────┐
Question ─> Surveyor ─> Planner ─> Orchestrator ─────────────┐ │ ─> Formatter ─> Consultation
                           ^       │          │              │ │
                           │   Researcher   Analyst          │ │
                           │       │          │              │ │
                           │       └──> Reviewer ────────────┘ │
                           │              │                    │
                           │        Critic (every 3 iter.)     │
                           │        │    │                     │
                           └────────┘  Adjudicator ────────────┘
                         └─────────────────────────────────────┘
```

An interactive D3.js version of this diagram is available on the [Architecture tab](https://agents.legal.org.ua) of the live demo.

## Agents

| # | Agent | When | What it does |
|---|-------|------|-------------|
| 1 | **Surveyor** | Once, at start | Maps the legal landscape: jurisdiction, applicable legislation, Supreme Court positions, risks |
| 2 | **Planner** | Once + on strategy critique | Produces and revises the research strategy and key legal questions |
| 3 | **Orchestrator** | Every iteration | Reads state, formulates hypotheses, dispatches tasks to Researcher or Analyst |
| 4 | **Researcher** | On dispatch | Searches case law, legislation, and doctrine via SecondLayer API (100M+ court decisions) |
| 5 | **Analyst** | On dispatch | Computes deadlines, penalties (Art. 549-552 CC), 3% annual (Art. 625 CC), inflation losses |
| 6 | **Reviewer** | After each research/analysis | Adversarial verification: checks citations exist, logic holds, evidence is current |
| 7 | **Critic** | Every 3 iterations | Senior legal advisor audit: strategy gaps, coherence, missing counterarguments |
| 8 | **Adjudicator** | On hypothesis conflict | Resolves disagreements between agents; can demote an established hypothesis back to working |
| 9 | **Formatter** | Once, after loop | Produces the final structured consultation with legislation references and case law citations |

## ConsultationState

All agents read and mutate a single state object:

| Field | Type | Description |
|-------|------|-------------|
| `client_question` | str | Original question |
| `jurisdiction` | str | civil, commercial, administrative, criminal |
| `survey_summary` | str | Legal landscape from Surveyor |
| `strategy` | LegalStrategy | Research plan from Planner |
| `hypotheses` | LegalHypothesis[] | Working/established/refuted legal positions |
| `evidence` | LegalEvidence[] | Case law, legislation, doctrine, computations |
| `questions` | ResearchQuestion[] | Open/resolved research questions |
| `critiques` | Critique[] | Active/resolved strategy and reasoning critiques |
| `answer` | str | Final formatted consultation |

## Critique Routing

When the Critic fires (every `critic_every_n` iterations), critiques are routed:

- **strategy** critique --> Planner (revise research strategy)
- **completeness** critique --> Orchestrator (generate new research questions)
- **hypothesis** critique --> Adjudicator (resolve conflict, may demote hypothesis)

## Project Structure

```
secondlayer-agents/
├── app.py                  # Gradio chat interface (prod + HF Space)
├── architecture.html       # D3.js interactive architecture diagram
├── src/lmaf/
│   ├── engine.py           # Main 4-phase pipeline loop (LMAF class)
│   ├── agents/             # 9 agent implementations (BaseAgent subclasses)
│   ├── state/              # ConsultationState, LoopState
│   ├── core/               # Config, WorkspaceManager (git-versioned)
│   ├── providers/          # LLM provider (AWS Bedrock)
│   ├── tools/              # SecondLayer API bridge (HTTP client)
│   ├── control/            # Loop control logic
│   ├── verification/       # Evidence verification utilities
│   └── rendering/          # State rendering for agent prompts
├── problems/               # Example legal problems (YAML)
├── tests/                  # pytest suite (state, config, app)
└── .github/workflows/      # CI/CD: test --> deploy prod + sync HF
```

## Quick Start

```bash
pip install -e .

# AWS Bedrock (required)
export AWS_REGION=eu-central-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# SecondLayer API (for case law and legislation search)
export SECONDLAYER_API_KEY=...

# Run a consultation
lmaf "Чи може продавець стягнути пеню за прострочення оплати товару?"

# Run from a problem file
lmaf problems/consumer_penalty.yaml
```

## Example Problems

| Problem | Domain | Question |
|---------|--------|----------|
| `consumer_penalty.yaml` | Civil | Penalty + 3% annual + inflation for late payment of 150K UAH goods |
| `labor_dismissal.yaml` | Labor | Challenging dismissal under Art. 40(1) during martial law |
| `property_dispute.yaml` | Family/Property | Property rights in an unregistered civil partnership (8 years) |

## CI/CD

Every push to `main` triggers:

1. **Test** -- `pytest` on ubuntu-latest (36 tests: state, config, app)
2. **Deploy to prod** -- SSH to prod, rebuild Docker container, health check
3. **Sync to HuggingFace** -- force-push to `overthelex/lmaf` Space

## Data Sources

| Source | What | Scale |
|--------|------|-------|
| [EDRSR](https://reyestr.court.gov.ua/) | Ukrainian court decisions | 100M+ documents |
| [Verkhovna Rada](https://zakon.rada.gov.ua/) | Ukrainian legislation | Full corpus |
| [ua-case-outcome-6m](https://huggingface.co/datasets/overthelex/ua-case-outcome-6m) | Court decisions dataset | 6.7M decisions |
| [ukrainian-court-decisions](https://huggingface.co/datasets/overthelex/ukrainian-court-decisions) | Balanced benchmark | 428K decisions |
| [edrsr-citation-graph-16m](https://huggingface.co/datasets/overthelex/edrsr-citation-graph-16m) | Citation graph | 16M edges |

## Related Papers

- [Temporal Decay of Co-Citation Predictability in Legal Statute Retrieval](https://arxiv.org/abs/2605.17639)
- [A Citation Graph from 100 Million Court Decisions](https://arxiv.org/abs/2605.15362)
- [Tokenizer Fertility and Zero-Shot Performance on Ukrainian Legal Text](https://arxiv.org/abs/2605.14890)

## License

Apache-2.0
