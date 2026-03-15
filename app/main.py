from dotenv import load_dotenv
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.core.limiter import limiter

load_dotenv()

from fastapi import FastAPI
from app.api.reflection_api import router as reflection_router

app = FastAPI(
    title="LifeOS API",
    description="AI powered life reflection system",
    version="1.0"
)

# 1. Attach the limiter state to the app
app.state.limiter = limiter

# 2. Add the exception handler for graceful 429 Too Many Requests errors
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(reflection_router, prefix="/api")