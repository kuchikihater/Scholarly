from typing import Annotated
import operator
import re
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict
from typing_extensions import List, TypedDict, Annotated, Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()


def initialization():
    class State(TypedDict):
        questions: Annotated[List[HumanMessage], operator.add]
        llm_responses: Annotated[List[Any], operator.add]
        summary: str
        qa_list: list
        response: str
    
    def extract_json_output(response: str):
        json_match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
        json_string = json_match.group(1).strip()
        parsed_json = json.loads(json_string)
        return parsed_json

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
            
            Then, tell the User that they can ask further questions about your final decision.
            """
        )

        summary = state["summary"]
        qa_list = state["qa_list"]

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"summary": summary, "qa_list": qa_list})
        return {"response": response}
    
    def follow_up_question(state: State):
        prompt = PromptTemplate.from_template(
            """
            You are an AI Assistant that just decided whether a paper should be published or not. 
            Your decision was based on:
            1) The provided summary of the paper.
            2) A list of questions and answers related to the paper.

            Here is the summary of the paper:
            {summary}
            
            Here is the Q&A list about the paper:
            {qa_list}

            Here is your previous response:
            {prev_response}
            
            Here is the user question:
            {query}

            First, determine whether the user question is relevant to your previously made final feedback or not.

            If the user question is relegvant to your final feedback, analyse the summary and Q&A List once more and give a fitting response, explaining your decisionmaking.
            If the user question is not relevant to your final feedback, tell the user politely to ask a more relevant question.
            """
        )

        summary = state["summary"]
        qa_list = state["qa_list"]
        prev_response = state ["response"]
        query = state["questions"][-1]

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"summary": summary, "qa_list": qa_list, "query": query, "prev_response": prev_response})
        
        return {"response": response}
    
    def more_questions_or_not(state: State):
        prompt = PromptTemplate.from_template(
            """
            Your task is to determine, if the user wants to ask further follow up questions or not.
            Evaluate the user input. If it says something along the lines of "no thank you", answer with "no".
            If the input is a question, answer "yes".
            Here is the user question:
            <question>
            {query}
            </question>
            After your assessment, provide your decision in JSON format. The JSON must contain a single key "feedback" with a value of either "yes" or "no". For example:

            {{
              "feedback": "yes"
            }}

            or

            {{
              "feedback": "no"
            }}

            Wrap your answer in <output> tag.
            """
        )

        chain = prompt | llm | StrOutputParser()
        question = state["questions"][-1]

        response = chain.invoke({"query": question})
        response_json = extract_json_output(response)

        if "yes" == response_json["feedback"]:
            return "yes"
        if "no" == response_json["feedback"]:
            return "no"


    graph_builder.add_node("followup questions", follow_up_question)
    graph_builder.add_node("chatbot", chatbot)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", "followup questions")
    graph_builder.add_conditional_edges(
        "followup questions",
        more_questions_or_not,
        {"yes": "followup questions", "no": END},
    )
    graph = graph_builder.compile()

    return graph
