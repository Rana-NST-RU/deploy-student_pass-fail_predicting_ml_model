import os
from typing import List
from tavily import TavilyClient


class WebSearchTool:
    def __init__(self, api_key: str = None):
        self.client = TavilyClient(api_key=api_key or os.getenv("TAVILY_API_KEY"))

    def search(self, query: str, max_results: int = 3) -> List[str]:
        try:
            results = self.client.search(query)
            formatted = []
            for r in results["results"][:max_results]:
                title = r.get("title", "Resource")
                url = r.get("url", "")
                formatted.append(f"{title} — {url}")
            return formatted
        except Exception as e:
            print(f"Web search failed: {e}")
            return []
