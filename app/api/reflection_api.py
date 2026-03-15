from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.limiter import limiter
from app.models.schemas import ReflectionRequest, ReflectionResponse
from app.agents.reflection_agent import analyze_day, analyze_day_stream

router = APIRouter()

@limiter.limit("3/minute")  # Set your limit here! (e.g., 3 requests per minute)
@router.post("/reflect", response_model=ReflectionResponse)
async def reflect_on_day(request: ReflectionRequest):
    """
        Takes the user's daily narration and runs it through the
        Orchestrator Agent to extract scores and actionable insights.
        """
    try:
        # Pass the narration string to your multi-agent system
        result = await analyze_day(request.narration)

        # FastAPI will automatically validate 'result' against ReflectionResponse
        # and serialize it into clean JSON for your frontend.
        return result

    except Exception as e:
        # Catch any LLM or ADK execution errors
        raise HTTPException(status_code=500, detail=f"Reflection analysis failed: {str(e)}")

# Add the new Streaming endpoint
@limiter.limit("3/minute")  # Set your limit here! (e.g., 3 requests per minute)
@router.post("/reflect/stream")
async def reflect_on_day_stream(request: ReflectionRequest):
    """
    Streams the execution progress of the multi-agent pipeline
    and returns the final JSON payload at the end.
    """
    try:
        # We pass the generator function directly to StreamingResponse
        # media_type="application/x-ndjson" tells the client to expect multiple JSON lines
        return StreamingResponse(
            analyze_day_stream(request.narration),
            media_type="application/x-ndjson"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")