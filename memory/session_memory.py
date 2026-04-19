from typing import Dict, List


class SessionMemory:
    """Keyed in-memory store for per-student chat history and metadata.

    Keys are arbitrary string IDs (e.g. "active" for the current session).
    """

    def __init__(self):
        self._store: Dict[str, dict] = {}

    def save(self, student_id: str, data: dict):
        """Merge *data* into the student's record (history is preserved)."""
        if student_id not in self._store:
            self._store[student_id] = {"history": []}
        self._store[student_id].update({k: v for k, v in data.items() if k != "history"})

    def load(self, student_id: str) -> dict:
        """Return the full record for *student_id*, or an empty dict."""
        return self._store.get(student_id, {})

    def get_history(self, student_id: str) -> List[dict]:
        """Return the conversation history list for *student_id*."""
        return self._store.get(student_id, {}).get("history", [])

    def set_history(self, student_id: str, history: List[dict]) -> None:
        """Replace the entire history list for *student_id*."""
        if student_id not in self._store:
            self._store[student_id] = {}
        self._store[student_id]["history"] = list(history)

    def add_turn(self, student_id: str, role: str, content: str):
        """Append a single {role, content} turn to *student_id*'s history."""
        if student_id not in self._store:
            self._store[student_id] = {"history": []}
        self._store[student_id]["history"].append({"role": role, "content": content})

    def clear(self, student_id: str) -> None:
        """Delete all stored data for *student_id*."""
        self._store.pop(student_id, None)
