from fastmcp import FastMCP, Tool
from game_core import GameCompanion

mcp = FastMCP()
companion = GameCompanion()


@mcp.tool(
    name="query_lore",
    description="Query Elden Ring lore and tactical information",
    parameters={
        "query": {
            "type": "string",
            "description": "The question or query about Elden Ring",
            "required": True,
        }
    },
)
def query_lore(query: str) -> str:
    return companion.query_lore(query)


if __name__ == "__main__":
    mcp.serve()