import os
from typing import Optional

import requests  # calling REST api
import tiktoken
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool  # setup tools for Agents
from langchain_community.vectorstores.faiss import FAISS  # vector database
from langchain_core.documents import Document  # wrap google search result
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # using OPENAI models
from langchain_openai import OpenAIEmbeddings
# langsmith tracing dashboard
from langsmith import traceable


class Model:
    _instance = None
    """Singleton Class that creates, runs agent and keep the memory update!"""

    def __new__(cls, api_key: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls)
            cls._instance.init(api_key)
        return cls._instance

    def init(self, api_key: Optional[str] = None):
        # Auto-trace LLM calls in-context
        self.tokens_count = 0
        self.memory = ConversationBufferMemory()
        self.api_key = api_key
        start_message = (
            "Hello! I am a helpful chatbot that answer questions about cars!"
        )
        self.memory.chat_memory.add_ai_message(start_message)
        self.prompt = hub.pull("hwchase17/react-chat")
        Chatllm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
        )
        self.tools = all_tools()
        self.agent = create_react_agent(llm=Chatllm, tools=self.tools, prompt=self.prompt)  # type: ignore
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, return_intermediate_steps=True)  # type: ignore

    @traceable(run_type="chain", tags=["dev"], project_name="Car Chatbot")
    def invoke(self, query: str):
        """run agent"""
        res = self.agent_executor(
            {"input": query, "chat_history": self.memory.load_memory_variables({})}
        )
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(res["output"])
        self.tokens_count += self.num_tokens_from_string(str(res))  # update tokens
        return res

    def get_memory(self):
        """return memory"""
        return self.memory.load_memory_variables({})

    def num_tokens_from_string(
        self, string: str, encoding_name: str = "gpt-3.5-turbo"
    ) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def get_tokens_count(self):
        """return tokens count"""
        return self.tokens_count



def all_tools():
    """return all tools"""
    return [get_luxury_car_reviews, google_search]


@tool
def get_luxury_car_reviews(query):
    """Useful to the real world reviews for the luxury cars"""
    # using the the same embedding vector that i used before.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        "./myindex",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.similarity_search(query, 5)


@tool
def google_search(query: str):
    """Useful to search any information relevant to cars."""

    # call google custom search api
    url = "https://customsearch.googleapis.com/customsearch/v1"
    args = {"key": os.environ.get("GOOGLE_API_KEY"), "cx": "e79101a8921564bb3"}
    args["q"] = query
    response = requests.get(url, params=args)
    metadata = {"source": "google_search"}
    # get only title and snippet of the search results
    res = ""
    if response.status_code == 200:
        data = response.json()
        for item in data.get("items"):
            res += item.get("title") + "\n"
            res += item.get("snippet") + "\n\n"
        return Document(page_content=res, metadata=metadata)
    return Document(
        page_content=f"Failed to retrieve data: {response.status_code}",
        metadata=metadata,
    )
