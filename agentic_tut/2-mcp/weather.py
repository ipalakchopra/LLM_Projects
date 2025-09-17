from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location:str)->str:
    """tells the weather at location"""
    return "It's always sunny in philadelphia"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")