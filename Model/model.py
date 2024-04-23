# import os

# import requests  # calling REST api
# from langchain import hub
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain.memory import ConversationBufferMemory
# from langchain.tools import tool  # setup tools for Agents
# from langchain_community.vectorstores.faiss import FAISS  # vector database
# from langchain_core.documents import Document  # wrap google search result
# from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI  # using OPENAI models
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# from Model import all_tools

# class Model():
#     """Class that creates, runs agent and keep the memory update!"""
    
#     def __init__(self,api_key):
#         print(f"Load environment: {load_dotenv()}")
#         self.memory = ConversationBufferMemory()
#         start_message = "Hello! I am a helpful chatbot that answer questions about cars!"
#         self.memory.chat_memory.add_ai_message(start_message)
#         self.prompt = hub.pull("hwchase17/react-chat")
#         chatllm = ChatOpenAI(
#             temperature=0,
#             api_key=api_key,
#             model="gpt-3.5-turbo",
#         )
#         self.tools = all_tools()
#         self.agent = create_react_agent(llm=chatllm, tools=self.tools, prompt=self.prompt) # type: ignore
#         self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True,return_intermediate_steps=True) # type: ignore
        
#     def invoke(self,query:str):
#         """run agent"""
#         res = self.agent_executor({"input":query,
#                 "chat_history": self.memory.load_memory_variables({})})
#         self.memory.chat_memory.add_user_message(query)
#         self.memory.chat_memory.add_ai_message(res['output'])
#         return res
        
#     def get_memory(self):
#         """return memory"""
#         return self.memory
        