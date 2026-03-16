from pydantic import BaseModel
from typing import Optional, List

class LifeScores(BaseModel):
    body: int
    mind: int
    emotion: int
    energy: int

class PastContext(BaseModel):
    core_identity: str
    recent_scores: Optional[List[LifeScores]] = []

class ReflectionRequest(BaseModel):
    narration: str
    historical_context: Optional[PastContext] = None

class ReflectionResponse(BaseModel):
    summary: str
    insights: List[str]
    scores: LifeScores

# --- Intermediate Schemas for Strict JSON Mode ---
class AgentInsights(BaseModel):
    insights: list[str]

class AgentSummary(BaseModel):
    summary: str
