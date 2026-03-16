import uuid
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from app.core.parser import JSONParser


class AgentExecutor:
    """
    Single Responsibility: Handles the network execution and session state of Google ADK Agents.
    Dependency Inversion: Decouples the Runner from the Business Logic.
    """

    def __init__(self):
        self.session_service = InMemorySessionService()

    async def execute(self, agent: LlmAgent, prompt_text: str) -> dict:
        # CRITICAL FIX: Generate a unique session ID for every execution to prevent collisions!
        unique_session_id = f"{agent.name}_session_{uuid.uuid4().hex}"

        session = await self.session_service.create_session(
            app_name="lifeos_engine",
            user_id="default_user",
            session_id=unique_session_id
        )

        runner = Runner(app_name="lifeos_engine", agent=agent, session_service=self.session_service)
        content = types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])

        final_text = ""
        async for event in runner.run_async(user_id="default_user", session_id=session.id, new_message=content):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text
                break

        # Delegate parsing to the dedicated parser class
        return JSONParser.parse_llm_output(final_text, agent.name)