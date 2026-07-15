from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv

load_dotenv()



class AgentState(TypedDict):
    message : str
    ideas : List[str]

llm = ChatAnthropic(model="claude-haiku-4-5", max_tokens=512)

def return_ideas(state : AgentState) -> AgentState:
    response = llm.invoke([HumanMessage(content=state["message"])])
    return {"ideas": [response.content]}

graph = StateGraph(AgentState)
graph.add_node("ideas", return_ideas)
graph.add_edge(START, "ideas")
graph.add_edge("ideas", END)
agent = graph.compile()

user_input = input("Enter:")
result = agent.invoke({"message": user_input, "ideas": []})
print(result["ideas"])


    
