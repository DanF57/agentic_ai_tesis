# server/mcp_server.py
"""
MCP Server: expone herramientas de búsqueda web, vector search
y actualización de la base de conocimiento usando Streamable HTTP Transport.

Este servidor NO razona. Solo ejecuta herramientas.
"""

from fastmcp import FastMCP
from server.tools.websearch_tool import web_search
from server.tools.vector_tool import vector_search

mcp = FastMCP(
    name="StudentKnowledgeTools",
    host="0.0.0.0",
    port=8050
)

# Registrar herramientas MCP
mcp.register(web_search)
mcp.register(vector_search)

if __name__ == "__main__":
    print("🚀 MCP Tools server corriendo en http://localhost:8050")
    mcp.run()

