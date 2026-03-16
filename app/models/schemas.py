from pydantic import BaseModel, field_validator
from typing import Optional, List, Any

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

    # THE SHIELD: Intercept the 'insights' array before strict validation
    @field_validator('insights', mode='before')
    @classmethod
    def sanitize_insights(cls, v: Any) -> List[str]:
        # If the AI completely failed and didn't return a list,
        # return it as-is and let Pydantic throw its standard error.
        if not isinstance(v, list):
            return v

        clean_list = []
        for item in v:
            if isinstance(item, str):
                # Happy Path: The LLM followed instructions and gave us clean strings.
                clean_list.append(item)
            elif isinstance(item, dict):
                # Hallucination Path: The LLM wrapped the string in an object
                # e.g., {"insight": "You worked too hard today."}
                # We grab the first value inside that dictionary and save it.
                values = list(item.values())
                if values:
                    clean_list.append(str(values[0]))
            else:
                # If it hallucinated numbers or booleans in the array, force them to strings
                clean_list.append(str(item))

        return clean_list

# --- Intermediate Schemas for Strict JSON Mode ---
class AgentInsights(BaseModel):
    insights: list[str]

class AgentSummary(BaseModel):
    summary: str
