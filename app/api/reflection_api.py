import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

# Import the rate limiter we set up earlier
from app.core.limiter import limiter

# Import your updated schemas
from app.models.schemas import ReflectionRequest, ReflectionResponse

# Import your real multi-agent execution functions (Handles both DEV/Ollama and PROD/Gemini)
from app.agents.reflection_agent import analyze_day, analyze_day_stream

router = APIRouter()

# Read the environment mode (Defaults to PROD if not set)
ENV_MODE = os.getenv("ENVIRONMENT", "PROD").upper()

# --- Standard REST Endpoint ---
@router.post("/reflect", response_model=ReflectionResponse)
async def reflect_on_day(payload: ReflectionRequest):
    """
    Standard REST endpoint.
    Routes to the mock generator or the active LLM strategy based on the environment.
    """
    try:
        # This automatically uses Ollama if ENV=DEV, or Gemini if ENV=PROD
        response = await analyze_day(
            narration=payload.narration,
            historical_context=payload.historical_context
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reflection failed: {str(e)}")


# --- Streaming Endpoint (For your Expo App) ---
@router.post("/reflect/stream")
@limiter.limit("3/minute")  # Protects your API credits/compute
async def reflect_on_day_stream(request: Request, payload: ReflectionRequest):
    """
    Streaming NDJSON endpoint.
    Streams real-time status updates to the frontend using the active environment strategy.
    Limited to 3 requests per minute per IP.
    """
    try:
        # This automatically uses Ollama if ENV=DEV, or Gemini if ENV=PROD
        generator = analyze_day_stream(
            narration=payload.narration,
            historical_context=payload.historical_context
        )

        # Pass the generator function to StreamingResponse
        return StreamingResponse(
            generator,
            media_type="application/x-ndjson"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")