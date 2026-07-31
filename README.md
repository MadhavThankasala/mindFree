# Draft Room

A multi-agent creative development tool. You give it a rough idea; it develops, critiques, shapes, and visualises it — and checks in with you when it needs a call.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and IBM Bob.

---

## What it does

You type a brief — as rough as "a short film about loneliness in a crowded city". Draft Room runs a pipeline of agents that:

1. **Ideator** — turns your brief into a fleshed-out concept, in plain human language
2. **Continuity Checker** — makes sure the concept hasn't drifted from what you asked for
3. **Critic** — gives honest, direct feedback and decides whether it's ready or needs another pass
4. **Formatter** — shapes the accepted concept into a proper artifact (film treatment, story sketch, startup pitch, song outline, etc.)
5. **Image Prompter** — writes a detailed visual prompt for Midjourney, DALL-E, or Stable Diffusion

The pipeline loops (up to 3 times) until the critic accepts. The web UI pauses after the critic step so you can review and redirect before it continues.

## Creative Modes

Pick a mode before you run — it shapes how every agent thinks and what it produces:

| Mode | Output |
|------|--------|
| General | Titled concept, plain prose |
| Short Film | Logline + setup / turn / resolution |
| Short Story | Opening paragraph + story arc |
| Startup Pitch | Problem / insight / product / why now |
| Song | Mood + theme + verse direction + hook |

---

## Setup

```bash
# 1. Clone and enter the project
cd DraftRoom

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
cp .env.example .env
# Edit .env — ANTHROPIC_API_KEY is required; OPENAI_API_KEY enables DALL-E image generation
```

## Running

**Web UI (recommended):**
```bash
streamlit run app.py
```

**CLI:**
```bash
python main.py
```

## Project structure

```
Draft Room/
├── agents/
│   ├── llm_client.py          # Centralised LLM factory
│   ├── ideator.py             # Develops the concept
│   ├── continuitiy_checker.py # Brief alignment check + brief validation
│   ├── critic.py              # Honest feedback + accept/revise verdict
│   ├── formatter.py           # Shapes concept into mode-appropriate artifact
│   └── image_prompter.py      # Generates visual prompt (reference-image-aware)
├── graph/
│   ├── state.py               # CreativeState schema + make_initial_state()
│   └── creative_graph.py      # LangGraph pipeline definition
├── tests/
│   └── test_pipeline.py       # Unit tests (no API calls, all mocked)
├── app.py                     # Streamlit web UI
├── main.py                    # CLI entry point
└── requirements.txt
```

## Running tests

```bash
pytest tests/
```

All tests are mocked — no API calls, no keys needed.

## Dependencies

- `langgraph` — pipeline orchestration with human-in-the-loop support
- `langchain-anthropic` — Claude integration
- `streamlit` — web UI
- `openai` — DALL-E image generation (optional)
- `python-dotenv` — environment variable management
