from typing import Dict, List


class SessionMemory:
    def __init__(self):
        self._store: Dict[str, dict] = {}

    def save(self, student_id: str, data: dict):
        if student_id not in self._store:
            self._store[student_id] = {"history": []}
        self._store[student_id].update({k: v for k, v in data.items() if k != "history"})

    def load(self, student_id: str) -> dict:
        return self._store.get(student_id, {})

    def get_history(self, student_id: str) -> List[dict]:
        return self._store.get(student_id, {}).get("history", [])

    def add_turn(self, student_id: str, role: str, content: str):
        if student_id not in self._store:
            self._store[student_id] = {"history": []}
        self._store[student_id]["history"].append({"role": role, "content": content})
