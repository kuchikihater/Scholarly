from typing import Annotated

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()


def initialization():
    class State(TypedDict):
        summary: str
        qa_list: list
        response: str


    graph_builder = StateGraph(State)

    llm = ChatOpenAI(model="gpt-4o-mini")


    def chatbot(state: State):
        prompt = PromptTemplate.from_template(
            """
            You are an AI Assistant that decides whether a paper should be published or not. 
            Your decision is based on:
            1) The provided summary of the paper.
            2) A list of questions and answers related to the paper.
            
            You must output either "accept" or "reject" as the final decision. 
            And also give reasoning, why you decided so based on if answers on questions where positive or negative, also use summary 
            
            Here is the summary of the paper:
            {summary}
            
            Here is the Q&A list about the paper:
            {qa_list}
            
            Analyze the summary and Q&A, then produce your final decision.
            """
        )

        summary = state["summary"]
        qa_list = state["qa_list"]

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"summary": summary, "qa_list": qa_list})
        return {"response": response}


    graph_builder.add_node("chatbot", chatbot)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()

    return graph
