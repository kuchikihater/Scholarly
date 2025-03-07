from typing import Annotated
import operator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAI

from ..config import OPENAI_MODEL_GPT4O
from ..utils.helpers import extract_json_output

from dotenv import load_dotenv

load_dotenv()


def initialization():
    class State(TypedDict):
        questions: Annotated[List[HumanMessage], operator.add]
        summary: str
        qa_list: list
        response: str
        final_feedback: str


    graph_builder = StateGraph(State)
    llm = ChatOpenAI(model=OPENAI_MODEL_GPT4O)

    def answer_follow_up_question(state: State):
        prompt = PromptTemplate.from_template(
            """
            You are a scientific peer reviewer. The user wants to discuss a research paper before making a final decision.
            Use the provided summary and Q&A list to answer the user's questions. 
            It also can be final_feedback be provided, but if place between tags <final_feedback> is empty, DO NOT pay attention

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
        ff = state.get("final_feedback", "")

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"summary": summary, "qa_list": qa_list, "query": query})
        state["response"] = response
        
        return {"response": response}

    def should_generate_final_feedback(state: State):
        prompt = PromptTemplate.from_template(
            """
            Your task is to determine if the user wants to get final decision or feedback about acceptance of paper or ask
            follow-up questions

            User input:
            {query}
            
            If the user is about to make final decision or feedback, return:
            {{ "feedback": "yes" }}

            If the user question is something else, return:
            {{ "feedback": "no"}}

            

            Wrap your answer in <output> tags.
            """
        )

        query = state["questions"][-1]
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"query": query})
        response_json = extract_json_output(response)

        return "yes" if response_json["feedback"] == "yes" else "no"

    def generate_final_feedback(state: State):
        prompt = PromptTemplate.from_template(
            """
            You are an AI Assistant tasked with providing a recommendation on whether a scientific paper should be published.
            You take on the role of a subreviewer for a peer review of a scientific paper. The user is the main reviewer.
 
            Your recommendation should be based on:
            1) The provided summary of the paper.
            2) A list of questions and answers related to the paper, representing a conversation about the paper.

            Your response should be structured as follows:

            Recommended: [Your recommendation - either "Accept" or "Reject"]
            Conversation Summary: [Briefly summarize the key points, questions, and concerns raised in the Q&A interaction.]
            Reasoning: [Provide a clear and critical explanation of the reasoning behind your recommendation. 
            Refer to specific aspects of the paper summary and the Q&A list. Highlight both positive and negative points.]
            Improvements Before Publishing: [Provide a concise list of actionable suggestions for improving the paper 
            before publication. Focus on the most important areas for improvement. Use bullet points.
            Consider BOTH:
                a) Specific issues and concerns raised in the Q&A.
                b) General improvements related to the paper's structure, clarity, completeness, and presentation, based on the 
                paper summary.
            ]

            Here is the summary of the paper:
            {summary}
            
            Here is the Q&A list about the paper (representing a conversation):
            {qa_list}
            
            Analyze the provided information and produce your recommendation, conversation summary, reasoning, 
            and improvement suggestions.

            Finally, ask the main reviewer (the user) if they have any follow-up questions.
            """
        )

        summary = state["summary"]
        qa_list = state["qa_list"]

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"summary": summary, "qa_list": qa_list})
        state["final_feedback"] = response
        state["response"] = response

        return {"final_feedback": response, "response": response}

    graph_builder.add_node("Discuss Paper", answer_follow_up_question)
    graph_builder.add_node("Final Feedback Node", generate_final_feedback)

    graph_builder.add_conditional_edges(
        START,
        should_generate_final_feedback,
        {"no": "Discuss Paper", "yes": "Final Feedback Node"},
    )
    graph_builder.add_edge("Final Feedback Node", END)

    graph = graph_builder.compile()
    return graph