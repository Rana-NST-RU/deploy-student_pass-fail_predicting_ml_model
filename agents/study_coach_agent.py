from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from agents.state import AgentState
from agents.nodes import (
    RouterNode,
    DiagnosisNode,
    PlannerNode,
    ResourceRetrieverNode,
    ResponseGeneratorNode,
    MemoryNode,
)


def _route(state: AgentState) -> str:
    return "diagnose" if state.get("is_study_query") else "respond"


class StudyCoachAgent:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("router", RouterNode())
        g.add_node("diagnose", DiagnosisNode())
        g.add_node("plan", PlannerNode())
        g.add_node("retrieve", ResourceRetrieverNode())
        g.add_node("respond", ResponseGeneratorNode())
        g.add_node("memory", MemoryNode())

        g.set_entry_point("router")
        g.add_conditional_edges(
            "router",
            _route,
            {"diagnose": "diagnose", "respond": "respond"},
        )
        g.add_edge("diagnose", "plan")
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "respond")
        g.add_edge("respond", "memory")
        g.add_edge("memory", END)

        return g.compile()

    def run(self, state: AgentState) -> AgentState:
        return self.graph.invoke(state)

    def chat(
        self,
        user_message: str,
        student_data: dict,
        session_history: list,
    ) -> tuple[str, list]:
        state = AgentState(
            messages=[HumanMessage(content=user_message)],
            student_data=student_data,
            learning_gaps=[],
            study_plan=None,
            retrieved_resources=[],
            session_history=session_history,
            current_goal=None,
            is_study_query=False,
        )
        result = self.run(state)
        ai_reply = result["messages"][-1].content
        updated_history = result["session_history"]
        return ai_reply, updated_history
