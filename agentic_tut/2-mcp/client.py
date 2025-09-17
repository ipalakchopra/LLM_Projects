from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
import asyncio
from dotenv import load_dotenv
load_dotenv()

async def main():
    client = MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["mathserver.py"],
                "transport":"stdio",
            },
            "weather":{
                "url":"http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    import os
    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

    tools = await client.get_tools()

    model = ChatGroq(model="llama-3.1-8b-instant")

    agent = create_react_agent(
        model, tools
    )

    math_response = await agent.ainvoke(
        {"messages":[{"role":"user","content":"whats is the weather in philadelphia"}]}
    )

    print(math_response["messages"][-1].content)

asyncio.run(main())