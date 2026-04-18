from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    student_data: dict
    learning_gaps: List[str]
    study_plan: Optional[str]
    retrieved_resources: List[str]
    session_history: List[dict]
    current_goal: Optional[str]
    is_study_query: bool
