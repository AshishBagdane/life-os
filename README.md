# LifeOS

An AI-powered life reflection and coaching service built with FastAPI and Google ADK agents. Submit your daily journal narration and get back health scores, actionable insights, a summary, and multi-perspective decision coaching — all processed through a sequential agent pipeline.

## Features

- **Daily Reflection** — Scores your day across four dimensions: body, mind, emotion, and energy (1–10 each)
- **Coaching Insights** — Extracts 2–3 concise, actionable behavioral insights from your narration
- **Day Summary** — Generates an empathetic 2–3 sentence synthesis of your day
- **Perspective Analysis** — Maps a decision or situation across multiple strategies with simulated outcomes, stakeholder mapping, and self-inquiry questions
- **Streaming** — Real-time NDJSON updates as each agent completes its step
- **Prompt Injection Defense** — User narration is always wrapped in XML tags; agent system prompts are hardened against override attempts
- **Dual-mode LLM** — Local Ollama in DEV, Google Gemini in PROD; swap with one env variable

## Tech Stack

| Layer | Library |
|---|---|
| API framework | FastAPI 0.115+ |
| AI agents | Google ADK 1.0+ |
| LLM (PROD) | Gemini 2.5 Flash |
| LLM (DEV) | Ollama via LiteLLM (`gemma3:1b`) |
| Schema / validation | Pydantic v2 |
| Rate limiting | SlowAPI |
| Server | Uvicorn |
| Python | ≥ 3.12 |

## Prerequisites

- Python 3.12+
- For **DEV mode**: [Ollama](https://ollama.com) running locally with `gemma3:1b` pulled
- For **PROD mode**: A `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com)

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd life-os

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set ENVIRONMENT and GEMINI_API_KEY as needed

# 4. Start the server (hot-reload on port 8080)
python run.py
```

Interactive API docs are available at `http://localhost:8080/docs`.

## Using Swagger UI

FastAPI generates interactive API docs automatically. Once the server is running, open `http://localhost:8080/docs` in your browser.

### Trying an endpoint

1. Click any endpoint (e.g. **POST /api/reflect**) to expand it.
2. Click **Try it out** in the top-right of that panel.
3. Edit the request body in the text area. A minimal example:

```json
{
  "narration": "Went for a run this morning, had a stressful meeting, then relaxed with a book in the evening."
}
```

4. Click **Execute**.
5. Scroll down to the **Responses** section to see the HTTP status, response headers, and the full JSON body.

### Trying the streaming endpoint

The Swagger UI executes `/api/reflect/stream` as a regular HTTP request and displays the raw NDJSON lines in the response body. Each line is a self-contained JSON object — you can read them top to bottom to follow the pipeline's progress. For a true streaming experience in the browser, use `fetch` with `ReadableStream` or `curl --no-buffer` from the terminal:

```bash
curl -X POST http://localhost:8080/api/reflect/stream \
  -H "Content-Type: application/json" \
  -N \
  -d '{"narration": "Had a productive day with deep focus work and an evening walk."}'
```

### ReDoc (alternative docs)

FastAPI also exposes `http://localhost:8080/redoc` — a read-only, three-panel reference layout that is better for reading schema details and nested model definitions.

### Alternative: uvicorn directly

```bash
uvicorn app.main:app --reload --port 8080
```

### Alternative: uv (recommended for isolated environments)

```bash
uv sync
uv run python run.py
```

## Environment Variables

Copy `.env.example` to `.env` and set:

| Variable | Required | Description |
|---|---|---|
| `ENVIRONMENT` | Yes | `DEV` (local Ollama) or `PROD` (Gemini cloud) |
| `GEMINI_API_KEY` | PROD only | API key from Google AI Studio |

**DEV mode** runs entirely locally — no API key, no cloud calls, free iteration.  
**PROD mode** uses `gemini-2.5-flash` for high-fidelity responses.

## API Reference

All endpoints are under the `/api` prefix.

---

### `POST /api/reflect`

Standard blocking reflection. Runs the full scoring → insights → summary pipeline and returns when complete.

**Request body**

```json
{
  "narration": "Today I worked out for 45 minutes, felt anxious before my presentation, but it went well. Had a productive evening reading.",
  "historical_context": {
    "core_identity": "Ambitious professional balancing fitness and mental health",
    "recent_scores": [
      { "body": 7, "mind": 6, "emotion": 5, "energy": 8 }
    ]
  }
}
```

`historical_context` is optional. When provided, agents incorporate past identity and trends into their analysis.

**Response**

```json
{
  "summary": "Despite pre-presentation anxiety, today was a productive and balanced day with physical exercise and intellectual engagement in the evening.",
  "insights": [
    "Regular exercise correlates with your higher energy scores — maintain the morning workout habit.",
    "Pre-event anxiety resolved positively; consider journaling before high-stakes moments to ground yourself."
  ],
  "scores": {
    "body": 8,
    "mind": 7,
    "emotion": 6,
    "energy": 7
  }
}
```

---

### `POST /api/reflect/stream`

Same pipeline as `/reflect` but streams real-time NDJSON progress updates as each agent finishes. Rate limited to **3 requests/minute per IP**.

**Response stream** (`application/x-ndjson`)

```
{"status": "processing", "step": "Evaluating health scores..."}
{"status": "processing", "step": "Extracting coaching insights...", "partial_scores": {"body": 8, "mind": 7, "emotion": 6, "energy": 7}}
{"status": "processing", "step": "Drafting final summary..."}
{"status": "complete", "data": { ...full ReflectionResponse... }}
```

On error:
```
{"status": "error", "message": "Pipeline error.", "detail": "..."}
```

---

### `POST /api/perspectives`

Decision coaching endpoint. Analyzes a situation or dilemma through a 6-step coaching framework and returns structured strategy simulations. Rate limited to **2 requests/minute per IP**.

**Request body** — same shape as `/reflect`.

**Response**

```json
{
  "situation_summary": "You are weighing whether to raise a concern with your manager about workload distribution.",
  "stakeholders": ["You", "Your manager", "Teammates affected by the imbalance"],
  "strategies": [
    {
      "strategy_name": "Direct conversation",
      "best_outcome": "Manager appreciates transparency and redistributes work fairly.",
      "most_realistic_outcome": "Manager acknowledges the issue but changes happen gradually.",
      "worst_outcome": "Manager perceives you as a complainer and your relationship suffers."
    },
    {
      "strategy_name": "Do nothing / Observe",
      "best_outcome": "Situation resolves itself as the project winds down.",
      "most_realistic_outcome": "Burnout accumulates silently over the next few weeks.",
      "worst_outcome": "You miss the window to address it and performance dips."
    }
  ],
  "reflection_questions": [
    "What outcome are you most afraid of, and how likely is it really?",
    "What would you advise a close friend in the same situation?"
  ],
  "recommended_approach": "Schedule a calm, factual conversation with your manager, framing it around team effectiveness rather than personal frustration."
}
```

---

## Architecture

```
POST /api/reflect
      │
      ▼
reflection_api.py          (FastAPI router — validates input, enforces rate limits)
      │
      ▼
AIOrchestrator             (sequential pipeline coordinator)
      │
      ├── scoring_agent ──► AgentExecutor ──► Google ADK Runner ──► LLM
      │        └── LifeScores
      │
      ├── insight_agent ──► AgentExecutor ──► Google ADK Runner ──► LLM
      │        └── List[str] insights
      │
      └── summary_agent ──► AgentExecutor ──► Google ADK Runner ──► LLM
               └── str summary
                        │
                        ▼
                 ReflectionResponse (Pydantic)
```

Each agent step's output is passed as context into the next step's prompt.

### Key Components

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI app init, rate-limit error handler |
| `app/api/reflection_api.py` | Route definitions, request/response wiring |
| `app/services/orchestrator.py` | Sequential agent pipeline and streaming generator |
| `app/agents/registry.py` | Agent persona definitions with hardened system prompts |
| `app/core/executor.py` | Google ADK `Runner` wrapper; creates a fresh UUID session per call |
| `app/core/llm_strategy.py` | Reads `ENVIRONMENT` and returns Ollama or Gemini model |
| `app/core/parser.py` | Strips markdown fences; regex fallback for chatty LLM output |
| `app/core/limiter.py` | SlowAPI limiter keyed by remote IP |
| `app/models/schemas.py` | Pydantic request/response models with LLM output sanitizers |

### Session Isolation

`AgentExecutor` creates a new `InMemorySessionService` session with a UUID-based ID for every agent invocation. This prevents session state from bleeding across concurrent requests.

### LLM Strategy

`LLMStrategyProvider.get_model()` is called once at import time in `registry.py`. All four agents share the same resolved model object, so switching environments requires only changing `ENVIRONMENT` in `.env` and restarting the server.

### Prompt Injection Defense

User narration is always injected between `<narration>...</narration>` XML tags. Each agent's system prompt explicitly:

1. Treats the narration as **passive data only**
2. Instructs the LLM to ignore any commands, persona overrides, or questions found within it
3. Specifies a safe default output if the input is malicious or nonsensical

### JSON Parsing

`JSONParser.parse_llm_output()` handles LLMs that wrap their JSON in markdown fences or prose:

1. Strip backtick fences and try direct `json.loads()`
2. Fall back to regex `\{.*\}` extraction with `re.DOTALL`
3. Raise a descriptive `ValueError` if both strategies fail

### Rate Limiting

SlowAPI limits by remote IP address:

| Endpoint | Limit |
|---|---|
| `POST /api/reflect` | Unrestricted |
| `POST /api/reflect/stream` | 3 / minute |
| `POST /api/perspectives` | 2 / minute |

Exceeded limits return HTTP `429 Too Many Requests`.

## Project Structure

```
life-os/
├── app/
│   ├── main.py               # FastAPI app entry point
│   ├── api/
│   │   └── reflection_api.py # Route handlers
│   ├── agents/
│   │   └── registry.py       # LlmAgent definitions
│   ├── core/
│   │   ├── executor.py       # ADK Runner wrapper
│   │   ├── limiter.py        # SlowAPI limiter instance
│   │   ├── llm_strategy.py   # DEV/PROD model selector
│   │   └── parser.py         # JSON extraction from LLM text
│   ├── models/
│   │   └── schemas.py        # Pydantic models
│   └── services/
│       └── orchestrator.py   # Agent pipeline logic
├── run.py                    # Uvicorn entry point
├── pyproject.toml            # Project metadata and tool config
├── requirements.txt          # Pinned runtime dependencies
└── .env.example              # Environment variable template
```

## Development

```bash
# Lint
uv run ruff check .

# Type-check
uv run mypy .

# Run tests
uv run pytest
```

Ruff is configured with `E`, `F`, `I`, `UP`, `B`, and `SIM` rule sets at line length 100. Mypy runs in strict mode with Pydantic plugin enabled.
