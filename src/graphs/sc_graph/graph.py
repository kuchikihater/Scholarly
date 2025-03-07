from typing import Annotated

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI

from src.config import OPENAI_MODEL_GPT4O_MINI

from dotenv import load_dotenv

load_dotenv()


def initialization():
    class State(TypedDict):
        # Messages have the type "list". The `add_messages` function
        # in the annotation defines how this state key should be updated
        # (in this case, it appends messages to the list, rather than overwriting them)
        messages: Annotated[list, add_messages]


    graph_builder = StateGraph(State)

    llm = ChatOpenAI(model=OPENAI_MODEL_GPT4O_MINI)


    def chatbot(state: State):
        prompt = PromptTemplate.from_template(
            """
            You are an AI Assistant.
            
            ADD after each your response reminder to the user that you can answer on question about paper if he would upload that
    
            Here is the initial query from the user:
            <question>
            {query} 
            </question>
            """
        )

        question = state["messages"][-1]

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"query": question})
        return {"messages": [response]}


    graph_builder.add_node("chatbot", chatbot)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()

    return graph