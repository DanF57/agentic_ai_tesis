# server/tools/websearch_tool.py
"""
Herramienta MCP: Búsqueda Web
Esta herramienta será utilizada por el agente cuando requiera información actual.
"""

from fastmcp import tool
import requests
import os

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

@tool()
def web_search(query: str) -> dict:
    """
    Realiza una consulta web usando Serper.dev o cualquier API que prefieras.
    """
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
    payload = {"q": query}

    resp = requests.post(url, json=payload, headers=headers)
    return resp.json()

