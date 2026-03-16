from pydantic import BaseModel, field_validator
from typing import Optional, List, Any

# --- 1. Sub-Models ---
class LifeScores(BaseModel):
    body: int
    mind: int
    emotion: int
    energy: int

class PastContext(BaseModel):
    core_identity: str
    recent_scores: Optional[List[LifeScores]] = []

# V2 FEATURE: The Nested Strategy Model
class StrategySimulation(BaseModel):
    strategy_name: str
    best_outcome: str
    most_realistic_outcome: str
    worst_outcome: str

# --- 2. Request Models ---
class ReflectionRequest(BaseModel):
    narration: str
    historical_context: Optional[PastContext] = None

# --- 3. Response Models ---
class ReflectionResponse(BaseModel):
    summary: str
    insights: List[str]
    scores: LifeScores

    @field_validator('insights', mode='before')
    @classmethod
    def sanitize_insights(cls, v: Any) -> List[str]:
        if not isinstance(v, list): return v
        clean_list = []
        for item in v:
            if isinstance(item, str): clean_list.append(item)
            elif isinstance(item, dict):
                values = list(item.values())
                if values: clean_list.append(str(values[0]))
            else: clean_list.append(str(item))
        return clean_list

# V2 FEATURE: The updated Perspective Schema
class PerspectiveResponse(BaseModel):
    situation_summary: str
    stakeholders: List[str]
    strategies: List[StrategySimulation]  # Using the nested model here!
    reflection_questions: List[str]
    recommended_approach: str

    @field_validator('stakeholders', 'reflection_questions', mode='before')
    @classmethod
    def sanitize_lists(cls, v: Any) -> List[str]:
        if not isinstance(v, list): return v
        clean_list = []
        for item in v:
            if isinstance(item, str): clean_list.append(item)
            elif isinstance(item, dict):
                values = list(item.values())
                if values: clean_list.append(str(values[0]))
            else: clean_list.append(str(item))
        return clean_list