from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
#from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Simple API Server"
)

llm = OllamaLLM(model="gemma")
#llm = Ollama(model="gemma")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words")

add_routes(
    app,
    prompt1|llm,
    path="/essay"
)
add_routes(
    app,
    prompt2|llm,
    path="/poem"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)