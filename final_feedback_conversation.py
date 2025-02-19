from typing import Annotated
import operator
import re
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict, List, Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

def initialization():
    class State(TypedDict):
        questions: Annotated[List[HumanMessage], operator.add]
        summary: str
        qa_list: list
        response: str
        final_feedback: str
    
    def extract_json_output(response: str):
        json_match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
        json_string = json_match.group(1).strip()
        parsed_json = json.loads(json_string)
        return parsed_json

    graph_builder = StateGraph(State)
    llm = ChatOpenAI(model="gpt-4o-mini")

    def discuss_paper(state: State):
        prompt = PromptTemplate.from_template(
            """
            You are a scientific peer reviewer. The user wants to discuss a research paper before making a final decision.
            Use the provided summary and Q&A list to answer the user's questions.
            
            Here is the summary of the paper:
            {summary}
            
            Here is the Q&A list about the paper:
            {qa_list}
            
            Here is the user question:
            {query}
            """
        )

        summary = state["summary"]
        qa_list = state["qa_list"]
        query = state["questions"][-1]

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"summary": summary, "qa_list": qa_list, "query": query})
        state["response"] = response
        return {"response": response}

    def more_questions_or_not(state: State):
        prompt = PromptTemplate.from_template(
            """
            Your task is to determine if the user wants to ask further follow-up questions or if they are ready for the final feedback.
            
            User input:
            {query}
            
            If the user still has more questions, return:
            {{ "feedback": "yes" }}
            
            If the user is ready for the final decision, return:
            {{ "feedback": "no" }}
            
            Wrap your answer in <output> tags.
            """
        )

        query = state["questions"][-1]
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"query": query})
        response_json = extract_json_output(response)

        return "yes" if response_json["feedback"] == "yes" else "no"

    def final_feedback(state: State):
        prompt = PromptTemplate.from_template(
            """
            Now that all discussions have taken place, it's time to make a final decision.
            
            You are a scientific peer reviewer, more specific a subreviewer. The User is the main reviewer and your task is to assist them by giving a recommendation on wether the uploaded paper should be accepted or declined for publishing.
            Based on the paper summary and the Q&A list, provide a final recommendation on whether the paper should be accepted for publication or not.
            
            Your answer should contain a clear decision (Accept or Reject) and a justification based on the summary and the Q&A list.
            
            Here is the summary:
            {summary}
            
            Here is the Q&A list:
            {qa_list}
            """
        )

        summary = state["summary"]
        qa_list = state["qa_list"]

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"summary": summary, "qa_list": qa_list})
        state["final_feedback"] = response
        return {"final_feedback": response}

    graph_builder.add_node("discuss_paper", discuss_paper)
    graph_builder.add_node("final_feedback", final_feedback)

    graph_builder.add_edge(START, "discuss_paper")
    graph_builder.add_conditional_edges(
        "discuss_paper",
        more_questions_or_not,
        {"yes": "discuss_paper", "no": "final_feedback"},
    )
    graph_builder.add_edge("final_feedback", END)

    graph = graph_builder.compile()
    return graph
