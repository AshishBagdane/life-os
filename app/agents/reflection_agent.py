import json
import re

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from app.models.schemas import ReflectionResponse, LifeScores, PastContext, AgentInsights, AgentSummary

from app.core.llm_strategy import LLMStrategyProvider # <-- Import the strategy

# 1. Resolve the model dynamically at startup
active_model = LLMStrategyProvider.get_model()

# 2. Define the agents once, injecting the active_model
scoring_agent = LlmAgent(
    name="scoring_agent",
    model=active_model,  # Automatically becomes Gemini or Ollama!
    instruction="""You are an expert holistic health evaluator. 
    Analyze the user's narration and determine scores from 1 to 10 for: body, mind, emotion, and energy.
    Output strictly as JSON matching the schema.""",
    output_schema=LifeScores
)

insight_agent = LlmAgent(
    name="insight_agent",
    model=active_model,
    instruction="""You are a lifestyle coach. Read the user's daily narration, their health scores, and historical context.
    Extract exactly 2 to 3 brief, actionable insights. Output strictly as JSON matching the schema."""
    # (output_schema omitted here for brevity, keep your existing ones!)
)

summary_agent = LlmAgent(
    name="summary_agent",
    model=active_model,
    instruction="""You are a thoughtful reflection assistant. 
    Write a brief, empathetic summary (2-3 sentences max) of their day.
    Output strictly as JSON matching the schema."""
)

async def analyze_day(narration: str, historical_context: PastContext = None) -> ReflectionResponse:
    """
    Executes a strict sequential multi-agent pipeline with historical context injection.
    """
    session_service = InMemorySessionService()

    # Helper function to run an individual agent safely (Unified for all LLMs)
    async def execute_agent(agent: LlmAgent, prompt_text: str) -> dict:
        session = await session_service.create_session(
            app_name="lifeos_reflection",
            user_id="default_user",
            session_id=f"{agent.name}_session"
        )
        runner = Runner(app_name="lifeos_reflection", agent=agent, session_service=session_service)
        content = types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])

        final_text = ""
        async for event in runner.run_async(
                user_id="default_user",
                session_id=session.id,
                new_message=content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text
                break

        # --- THE UNIFIED PARSER ---

        # 1. First, try the clean Gemini way (Fast string stripping)
        clean_text = final_text.strip("`").removeprefix("json").strip()

        try:
            # If Gemini followed the rules, this works instantly.
            return json.loads(clean_text)

        except json.JSONDecodeError:
            # 2. If it fails (e.g., Ollama added conversational filler), fall back to Regex
            match = re.search(r'\{.*\}', final_text, re.DOTALL)

            if not match:
                print(f"🚨 Parsing completely failed for {agent.name}. Raw output:\n{final_text}")
                raise ValueError(f"No JSON block found in response from {agent.name}.")

            try:
                # Try parsing the Regex-extracted block
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                print(f"🚨 Regex extraction failed for {agent.name}. Extracted block:\n{match.group(0)}")
                raise e

    # --- Prepare Historical Context ---
    history_str = ""
    if historical_context:
        history_str = f"\nHistorical Identity: {historical_context.core_identity}\nRecent Scores: {[s.model_dump() for s in historical_context.recent_scores]}"

    # --- Pipeline Step 1: Get Scores (Objective, no history) ---
    score_data = await execute_agent(scoring_agent, f"Narration: {narration}")
    scores = LifeScores(**score_data)

    # --- Pipeline Step 2: Get Insights (Trend-aware) ---
    insight_prompt = f"Narration: {narration}\nEvaluated Scores: {scores.model_dump()}{history_str}"
    insight_data = await execute_agent(insight_agent, insight_prompt)

    # --- Pipeline Step 3: Get Summary (Deeply contextual) ---
    summary_prompt = f"Narration: {narration}\nScores: {scores.model_dump()}\nInsights: {insight_data['insights']}{history_str}"
    summary_data = await execute_agent(summary_agent, summary_prompt)

    # --- Final Step: Assemble the ReflectionResponse ---
    return ReflectionResponse(
        summary=summary_data["summary"],
        insights=insight_data["insights"],
        scores=scores
    )

async def analyze_day_stream(narration: str, historical_context: PastContext = None):
    """
    Executes the sequential pipeline with historical context and yields real-time progress updates.
    """
    session_service = InMemorySessionService()

    async def execute_agent(agent, prompt_text: str) -> dict:
        session = await session_service.create_session(
            app_name="lifeos_reflection",
            user_id="default_user",
            session_id=f"{agent.name}_session"
        )
        runner = Runner(app_name="lifeos_reflection", agent=agent, session_service=session_service)
        content = types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])

        final_text = ""
        async for event in runner.run_async(
                user_id="default_user",
                session_id=session.id,
                new_message=content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text
                break

        clean_json = final_text.strip("`").removeprefix("json").strip()
        return json.loads(clean_json)

    # --- Prepare Historical Context ---
    history_str = ""
    if historical_context:
        history_str = f"\nHistorical Identity: {historical_context.core_identity}\nRecent Scores: {[s.model_dump() for s in historical_context.recent_scores]}"

    # --- Step 1: Scoring ---
    yield json.dumps({"status": "processing", "step": "Evaluating health scores..."}) + "\n"
    score_data = await execute_agent(scoring_agent, f"Narration: {narration}")
    scores = LifeScores(**score_data)

    # --- Step 2: Insights ---
    yield json.dumps({
        "status": "processing",
        "step": "Extracting coaching insights...",
        "partial_scores": scores.model_dump()
    }) + "\n"

    insight_prompt = f"Narration: {narration}\nEvaluated Scores: {scores.model_dump()}{history_str}"
    insight_data = await execute_agent(insight_agent, insight_prompt)

    # --- Step 3: Summary ---
    yield json.dumps({"status": "processing", "step": "Drafting final summary..."}) + "\n"
    summary_prompt = f"Narration: {narration}\nScores: {scores.model_dump()}\nInsights: {insight_data['insights']}{history_str}"
    summary_data = await execute_agent(summary_agent, summary_prompt)

    # --- Final Step: Complete Payload ---
    final_response = ReflectionResponse(
        summary=summary_data["summary"],
        insights=insight_data["insights"],
        scores=scores
    )

    yield json.dumps({
        "status": "complete",
        "data": final_response.model_dump()
    }) + "\n"