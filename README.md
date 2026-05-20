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

A multi-agent framework for **complex legal consultations** over Ukrainian court decisions and legislation.

Built for the legal domain with access to 100M+ Ukrainian court decisions via [SecondLayer](https://legal.org.ua).

## Architecture

Nine specialised LLM agents work in a loop, each starting from a fresh context (no conversation history). All state lives in a structured `ConsultationState` object. The workspace is git-versioned for full reproducibility.

```
                    ┌─────────────┐
                    │   Client    │
                    │  Question   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Surveyor   │  Maps legal landscape (once)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Planner   │  Research strategy
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │      Orchestrator       │  Dispatches tasks
              └───┬─────────────────┬───┘
                  │                 │
           ┌──────▼──────┐  ┌──────▼──────┐
           │ Researcher  │  │   Analyst   │
           │ (case law,  │  │ (deadlines, │
           │ legislation)│  │  penalties) │
           └──────┬──────┘  └──────┬──────┘
                  │                 │
              ┌───▼─────────────────▼───┐
              │       Reviewer          │  Adversarial verification
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    Critic (periodic)    │  Strategy audit
              └──────┬──────────┬───────┘
                     │          │
              ┌──────▼──┐ ┌────▼────────┐
              │ Planner │ │ Adjudicator │  Resolve disputes
              │(revise) │ └─────────────┘
              └─────────┘
                           │
              ┌────────────▼────────────┐
              │       Formatter         │  Final consultation
              └─────────────────────────┘
```

### Agent Roles

| Agent | Role | secondlayer-core Equivalent |
|-------|------|---------------------------|
| **Surveyor** | Maps legal landscape, identifies relevant areas of law | IntentClassifier + QueryPlanner |
| **Planner** | Produces/revises research strategy | ExecutionPlan generation |
| **Orchestrator** | Dispatches tasks, formulates hypotheses | ChatService agentic loop |
| **Researcher** | Finds case law, legislation, doctrine via SecondLayer MCP | Tool calls (search, legislation) |
| **Analyst** | Computes deadlines, penalties, procedural checks | Calculation tool calls |
| **Reviewer** | Adversarial verification of evidence | CitationValidator + HallucinationGuard |
| **Critic** | Periodic strategy and coherence audit | *New capability* |
| **Adjudicator** | Resolves inter-agent disagreements | *New capability* |
| **Formatter** | Produces structured legal consultation | Response synthesis |

### Key Design Decisions

1. **No agent carries conversation history** -- each call starts from a fresh context with the current `ConsultationState` rendered as text. This prevents context contamination and allows any agent to be swapped or retried independently.

2. **Structured state** -- `ConsultationState` tracks hypotheses, evidence, critiques, and their relationships. Agents mutate state via typed operations, not free-form text.

3. **Git-versioned workspace** -- every iteration creates a commit, making the full research process reproducible and auditable.

4. **SecondLayer MCP bridge** -- agents access 100M+ court decisions, legislation, and Supreme Court positions through the SecondLayer API.

## Quick Start

```bash
# Install
pip install -e .

# Set API keys
export ANTHROPIC_API_KEY=sk-...
export SECONDLAYER_API_KEY=...

# Run a consultation
lmaf "Чи може продавець стягнути пеню за прострочення оплати товару?"

# Run from a problem file
lmaf problems/consumer_penalty.yaml
```

## Example Problems

| Problem | Difficulty | Description |
|---------|-----------|-------------|
| `consumer_penalty.yaml` | Medium | Penalty and interest calculation for late payment |
| `labor_dismissal.yaml` | Hard | Challenging unlawful dismissal during martial law |
| `property_dispute.yaml` | Hard | Property rights in unregistered partnership |

## Data Sources

- [EDRSR](https://reyestr.court.gov.ua/) -- 100M+ Ukrainian court decisions
- [Verkhovna Rada](https://zakon.rada.gov.ua/) -- Ukrainian legislation
- [overthelex/ua-case-outcome-6m](https://huggingface.co/datasets/overthelex/ua-case-outcome-6m) -- 6.7M court decisions dataset
- [overthelex/ukrainian-court-decisions](https://huggingface.co/datasets/overthelex/ukrainian-court-decisions) -- 428K balanced benchmark

## Related Papers

- [Temporal Decay of Co-Citation Predictability](https://arxiv.org/abs/2605.17639)
- [A Citation Graph from 100M Court Decisions](https://arxiv.org/abs/2605.15362)
- [Tokenizer Fertility and Zero-Shot Performance on Ukrainian Legal Text](https://arxiv.org/abs/2605.14890)

## License

Apache-2.0
