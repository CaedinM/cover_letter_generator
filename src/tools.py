import os
from tavily import TavilyClient
from crewai.tools import tool


@tool("Company Research Tool")
def search_company(query: str) -> str:
    """Search the web for information about a company, including their mission, values, culture, recent news, and products. Use this to research companies mentioned in job descriptions."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search unavailable - TAVILY_API_KEY not set."

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, search_depth="advanced", max_results=5)

    results = []
    if response.get("answer"):
        results.append(f"Summary: {response['answer']}\n")

    for r in response.get("results", []):
        results.append(f"- {r.get('title', 'No title')}: {r.get('content', 'No content')[:500]}")

    return "\n".join(results) if results else "No results found."