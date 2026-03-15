import json

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from app.models.schemas import ReflectionResponse, LifeScores, AgentSummary, AgentInsights

# --- 1. The Scoring Agent ---
scoring_agent = LlmAgent(
    name="scoring_agent",
    model="gemini-2.5-flash",
    instruction="""You are an expert holistic health evaluator. 
    Analyze the user's narration and determine scores from 1 to 10 for: body, mind, emotion, and energy.
    Output strictly as JSON matching the schema.""",
    output_schema=LifeScores  # Safe to use now! No sub_agents attached.
)

# --- 2. The Insight Agent ---
insight_agent = LlmAgent(
    name="insight_agent",
    model="gemini-2.5-flash",
    instruction="""You are a lifestyle coach. Read the user's narration and their health scores.
    Extract exactly 2 to 3 brief, actionable insights about how their actions affected their mood or health.
    Output strictly as JSON matching the schema.""",
    output_schema=AgentInsights
)

# --- 3. The Summary Agent ---
summary_agent = LlmAgent(
    name="summary_agent",
    model="gemini-2.5-flash",
    instruction="""You are a thoughtful reflection assistant. 
    Read the user's narration, their health scores, and the generated insights.
    Write a brief, empathetic summary (2-3 sentences max) of their day.
    Output strictly as JSON matching the schema.""",
    output_schema=AgentSummary
)

# --- The Sequential Execution Pipeline ---
async def analyze_day(narration: str) -> ReflectionResponse:
    """
    Executes a strict sequential multi-agent pipeline to guarantee structured output.
    """
    session_service = InMemorySessionService()

    # Helper function to run an individual agent safely
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

        # Because we use output_schema, the text is guaranteed to be clean JSON
        clean_json = final_text.strip("`").removeprefix("json").strip()
        return json.loads(clean_json)

    # --- Pipeline Step 1: Get Scores ---
    score_data = await execute_agent(scoring_agent, f"Narration: {narration}")
    scores = LifeScores(**score_data)

    # --- Pipeline Step 2: Get Insights ---
    insight_prompt = f"Narration: {narration}\nEvaluated Scores: {scores.model_dump()}"
    insight_data = await execute_agent(insight_agent, insight_prompt)

    # --- Pipeline Step 3: Get Summary ---
    summary_prompt = f"Narration: {narration}\nScores: {scores.model_dump()}\nInsights: {insight_data['insights']}"
    summary_data = await execute_agent(summary_agent, summary_prompt)

    # --- Final Step: Assemble the ReflectionResponse ---
    # We use standard Python to assemble the final object, ensuring 100% type safety.
    return ReflectionResponse(
        summary=summary_data["summary"],
        insights=insight_data["insights"],
        scores=scores
    )

async def analyze_day_stream(narration: str):
    """
    Executes the sequential pipeline and yields real-time progress updates.
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

    # --- Step 1: Scoring ---
    # Yield a status update to the frontend
    yield json.dumps({"status": "processing", "step": "Evaluating health scores..."}) + "\n"

    score_data = await execute_agent(scoring_agent, f"Narration: {narration}")
    scores = LifeScores(**score_data)

    # --- Step 2: Insights ---
    # Yield the next status, and optionally pass the completed scores early!
    yield json.dumps({
        "status": "processing",
        "step": "Extracting coaching insights...",
        "partial_scores": scores.model_dump()
    }) + "\n"

    insight_prompt = f"Narration: {narration}\nEvaluated Scores: {scores.model_dump()}"
    insight_data = await execute_agent(insight_agent, insight_prompt)

    # --- Step 3: Summary ---
    yield json.dumps({"status": "processing", "step": "Drafting final summary..."}) + "\n"

    summary_prompt = f"Narration: {narration}\nScores: {scores.model_dump()}\nInsights: {insight_data['insights']}"
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