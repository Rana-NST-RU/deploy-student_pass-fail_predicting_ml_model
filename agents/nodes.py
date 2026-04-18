import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agents.state import AgentState
from tools.web_search_tool import WebSearchTool
from tools.rag_tool import RAGTool

_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)


class RouterNode:
    def __call__(self, state: AgentState) -> AgentState:
        last_msg = state["messages"][-1].content
        system = (
            "Classify this student query. Reply with exactly one word only: "
            "STUDY if it's about academic performance, study plans, scores, learning gaps, or study advice. "
            "GENERAL if it's casual conversation, greetings, or unrelated to studying."
        )
        result = _llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=last_msg)]
        )
        state["is_study_query"] = "STUDY" in result.content.upper()
        return state


class DiagnosisNode:
    def __call__(self, state: AgentState) -> AgentState:
        data = state["student_data"]
        gaps = []
        if float(data.get("math_score", 100)) < 50:
            gaps.append("math (score below 50)")
        if float(data.get("reading_score", 100)) < 50:
            gaps.append("reading (score below 50)")
        if float(data.get("writing_score", 100)) < 50:
            gaps.append("writing (score below 50)")
        if float(data.get("attendance_rate", 1.0)) < 0.75:
            gaps.append("attendance (below 75%)")
        if float(data.get("daily_study_hours", 5)) < 2:
            gaps.append("study time (less than 2 hours/day)")
        if float(data.get("stress_level", 5)) > 7:
            gaps.append("stress management (high stress level)")
        if float(data.get("sleep_hours", 8)) < 6:
            gaps.append("sleep (less than 6 hours/night)")
        if float(data.get("motivation_score", 50)) < 30:
            gaps.append("motivation (low score)")
        state["learning_gaps"] = gaps if gaps else ["no major gaps detected"]
        return state


class PlannerNode:
    def __call__(self, state: AgentState) -> AgentState:
        data = state["student_data"]
        gaps = state["learning_gaps"]
        prediction = data.get("prediction", "unknown")
        prompt = (
            f"You are a study coach. A student's predicted outcome is '{prediction}'.\n"
            f"Their weak areas: {', '.join(gaps)}.\n"
            f"Their profile: daily study hours={data.get('daily_study_hours')}, "
            f"attendance rate={data.get('attendance_rate')}, "
            f"stress level={data.get('stress_level')}, "
            f"sleep hours={data.get('sleep_hours')}, "
            f"math score={data.get('math_score')}, "
            f"reading score={data.get('reading_score')}, "
            f"writing score={data.get('writing_score')}.\n"
            "Create a concise, actionable 7-day study plan targeting these weak areas. "
            "Be specific with daily tasks. Format as Day 1 through Day 7."
        )
        result = _llm.invoke([HumanMessage(content=prompt)])
        state["study_plan"] = result.content
        return state


class ResourceRetrieverNode:
    def __init__(self):
        self.rag = RAGTool()
        self.search = WebSearchTool()

    def __call__(self, state: AgentState) -> AgentState:
        gaps = state["learning_gaps"]
        query = f"study tips and resources for {', '.join(gaps[:2])}"
        rag_results = self.rag.retrieve(query, k=3)
        web_results = self.search.search(query, max_results=2)
        state["retrieved_resources"] = rag_results + web_results
        return state


class ResponseGeneratorNode:
    def __call__(self, state: AgentState) -> AgentState:
        last_msg = state["messages"][-1].content
        history = state.get("session_history", [])[-6:]
        history_text = "\n".join(
            [f"{t['role'].capitalize()}: {t['content']}" for t in history]
        )

        if state.get("is_study_query"):
            resources_preview = "\n".join(
                [f"- {r[:120]}" for r in state.get("retrieved_resources", [])[:3]]
            )
            system = (
                "You are a friendly, encouraging AI study coach.\n"
                f"Student data: {state['student_data']}\n"
                f"Learning gaps identified: {', '.join(state.get('learning_gaps', []))}\n"
                f"Generated study plan:\n{state.get('study_plan', 'Not generated')}\n"
                f"Relevant resources:\n{resources_preview}\n"
                f"Conversation so far:\n{history_text}\n\n"
                "Respond helpfully and conversationally. Reference the student's specific data "
                "when relevant. Keep responses concise and encouraging."
            )
        else:
            system = (
                "You are a friendly AI study coach having a general conversation with a student.\n"
                f"Conversation so far:\n{history_text}\n\n"
                "Respond naturally, warmly, and helpfully."
            )

        result = _llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=last_msg)]
        )
        state["messages"] = [AIMessage(content=result.content)]
        return state


class MemoryNode:
    def __call__(self, state: AgentState) -> AgentState:
        msgs = state["messages"]
        last_human = next(
            (m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), ""
        )
        last_ai = next(
            (m.content for m in reversed(msgs) if isinstance(m, AIMessage)), ""
        )
        history = list(state.get("session_history", []))
        if last_human:
            history.append({"role": "user", "content": last_human})
        if last_ai:
            history.append({"role": "assistant", "content": last_ai})
        state["session_history"] = history
        return state
