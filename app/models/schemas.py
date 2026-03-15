from pydantic import BaseModel

class ReflectionRequest(BaseModel):
    narration: str

class LifeScores(BaseModel):
    body: int
    mind: int
    emotion: int
    energy: int

class ReflectionResponse(BaseModel):
    summary: str
    insights: list[str]
    scores: LifeScores

# --- Intermediate Schemas for Strict JSON Mode ---
class AgentInsights(BaseModel):
    insights: list[str]

class AgentSummary(BaseModel):
    summary: str
