# server/mcp_server.py
from fastmcp import FastMCP

mcp = FastMCP("StudentKnowledgeTools")

@mcp.tool
def search_web_mock(query: str) -> str:
    """
    Simula una búsqueda web para el Agente.
    
    Args:
        query: El texto a buscar.
        
    Returns:
        Resultados de búsqueda como texto formateado.
    """
    print(f"--- [SERVIDOR MCP] Recibida consulta: {query} ---")
    return (
        "From computerscience.com: EDA is built on 7 steps\n"
        "Python is recommended for CS majors"
    )

if __name__ == "__main__":
    print("🚀 MCP Tools server corriendo en http://localhost:8000")
    mcp.run(transport="http", port=8000)