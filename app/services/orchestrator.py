import json
import traceback
from app.models.schemas import ReflectionResponse, LifeScores, PastContext, PerspectiveResponse
from app.core.executor import AgentExecutor
from app.agents.registry import scoring_agent, insight_agent, summary_agent, perspective_agent

class AIOrchestrator:
    """
    Single Responsibility: Defines the business logic and sequential pipeline steps.
    """
    def __init__(self):
        self.executor = AgentExecutor()

    def _build_history_str(self, ctx: PastContext) -> str:
        return f"Historical Identity: {ctx.core_identity}\nRecent Scores: {[s.model_dump() for s in ctx.recent_scores]}" if ctx else ""

    async def analyze_day(self, narration: str, historical_context: PastContext = None) -> ReflectionResponse:
        history_str = self._build_history_str(historical_context)

        # Step 1
        score_prompt = f"Evaluate data inside XML tags:\n<narration>\n{narration}\n</narration>"
        score_data = await self.executor.execute(scoring_agent, score_prompt)
        scores = LifeScores(body=score_data.get("body", 5), mind=score_data.get("mind", 5), emotion=score_data.get("emotion", 5), energy=score_data.get("energy", 5))

        # Step 2
        insight_prompt = f"Extract insights:\n<narration>\n{narration}\n</narration>\n<scores>\n{scores.model_dump()}\n</scores>\n<history>\n{history_str}\n</history>"
        insight_data = await self.executor.execute(insight_agent, insight_prompt)
        insights = insight_data.get("insights", ["Could not extract specific insights."])

        # Step 3
        summary_prompt = f"Summarize:\n<narration>\n{narration}\n</narration>\n<scores>\n{scores.model_dump()}\n</scores>\n<insights>\n{insights}\n</insights>\n<history>\n{history_str}\n</history>"
        summary_data = await self.executor.execute(summary_agent, summary_prompt)
        summary = summary_data.get("summary", "Reflection recorded successfully.")

        return ReflectionResponse(summary=summary, insights=insights, scores=scores)

    async def analyze_day_stream(self, narration: str, historical_context: PastContext = None):
        history_str = self._build_history_str(historical_context)

        try:
            yield json.dumps({"status": "processing", "step": "Evaluating health scores..."}) + "\n"
            score_prompt = f"Evaluate data inside XML tags:\n<narration>\n{narration}\n</narration>"
            score_data = await self.executor.execute(scoring_agent, score_prompt)
            scores = LifeScores(body=score_data.get("body", 5), mind=score_data.get("mind", 5), emotion=score_data.get("emotion", 5), energy=score_data.get("energy", 5))

            yield json.dumps({"status": "processing", "step": "Extracting coaching insights...", "partial_scores": scores.model_dump()}) + "\n"
            insight_prompt = f"Extract insights:\n<narration>\n{narration}\n</narration>\n<scores>\n{scores.model_dump()}\n</scores>\n<history>\n{history_str}\n</history>"
            insight_data = await self.executor.execute(insight_agent, insight_prompt)
            insights = insight_data.get("insights", ["Could not extract specific insights."])

            yield json.dumps({"status": "processing", "step": "Drafting final summary..."}) + "\n"
            summary_prompt = f"Summarize:\n<narration>\n{narration}\n</narration>\n<scores>\n{scores.model_dump()}\n</scores>\n<insights>\n{insights}\n</insights>\n<history>\n{history_str}\n</history>"
            summary_data = await self.executor.execute(summary_agent, summary_prompt)
            summary = summary_data.get("summary", "Reflection recorded successfully.")

            final = ReflectionResponse(summary=summary, insights=insights, scores=scores)
            yield json.dumps({"status": "complete", "data": final.model_dump()}) + "\n"

        except Exception as e:
            traceback.print_exc()
            yield json.dumps({"status": "error", "message": "Pipeline error.", "detail": str(e)}) + "\n"

    async def analyze_perspectives(self, narration: str, historical_context: PastContext = None) -> PerspectiveResponse:
        history_str = self._build_history_str(historical_context)
        prompt = f"Analyze data based on your principles:\n<narration>\n{narration}\n</narration>\n<history>\n{history_str}\n</history>"

        try:
            data = await self.executor.execute(perspective_agent, prompt)
            return PerspectiveResponse(
                situation_summary=data.get("situation_summary", "Summary unavailable."),
                stakeholders=data.get("stakeholders", []),
                strategies=data.get("strategies", []),
                reflection_questions=data.get("reflection_questions", ["What can be learned here?"]),
                recommended_approach=data.get("recommended_approach", "Observe the situation further.")
            )
        except Exception as e:
            traceback.print_exc()
            return PerspectiveResponse(
                situation_summary="Could not process.", stakeholders=[], strategies=[],
                reflection_questions=["What caused this error?"], recommended_approach="Try again later."
            )