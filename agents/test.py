import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor, initialize_agent, AgentType
from langchain.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.vectorstores import FAISS
#from langchain_openai import OpenAIEmbeddings  # Using OpenAI for embedding generation, if needed
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama  # Use Ollama API for language model interaction
from langchain_community.embeddings import OllamaEmbeddings

# Load environment variables from .env
load_dotenv()

# Set Ollama API Key if needed
#os.environ['OLLAMA_API_KEY'] = os.getenv("OLLAMA_API_KEY")

# Function to generate responses using Ollama's LLaMA 2 model
def generate_llama2_response(input_text):
    # Use the Ollama API to generate responses
    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": input_text}])
    generated_text = response['text']  # Extract the text from the response
    return generated_text

# Setup Wikipedia and Arxiv API Wrappers
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)

# Setup tools for query processing
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

# Setting up vector store for LangSmith search using FAISS
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents, OllamaEmbeddings())
retriever = vectordb.as_retriever()

# Create retriever tool for LangSmith search
from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(retriever, "langsmith_search", 
                                      "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

# Combine tools
tools = [wiki, arxiv, retriever_tool]

# Define LLaMA 2 powered agent using LangChain
llm = generate_llama2_response  # Use LLaMA 2 function from Ollama

# Load a prompt template (customize as needed)
prompt = hub.pull("hwchase17/openai-functions-agent")
print(prompt.messages)

# Create the agent using the tools
#agent = create_openai_tools_agent(llm, tools, prompt)
agent = initialize_agent(
    tools=tools,
    llm=generate_llama2_response,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # You can experiment with other agent types too
    verbose=True
)


# Set up the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
response = agent_executor.invoke({"input": "Tell me about Langsmith"})
print(response)

# Example: Use Arxiv tool to query scientific paper
arxiv_query_response = agent_executor.invoke({"input": "Tell me about arxiv paper 1605.08386"})
print(arxiv_query_response)
