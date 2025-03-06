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
        flag: int 

    def extract_json_output(response: str):
        json_match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
        json_string = json_match.group(1).strip()
        parsed_json = json.loads(json_string)
        return parsed_json

    graph_builder = StateGraph(State)
    llm = ChatOpenAI(model="gpt-4o")

    def check_generation_feedback(state: State):
        response = "yes" if state.get("flag", 0) == 1 else "no"
        return {"response": response}
    

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
        response = "yes" if response_json["feedback"] == "yes" else "no"
        return {"response": response}

    def final_feedback(state: State):
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
        state["flag"] = 1  
        return {"final_feedback": response, "response": response}
    
    def follow_up_questions(state: State):
        prompt = PromptTemplate.from_template(
            """
            You are a subreviewer for a peer reviewing of a research paper continuing a scientific peer review discussion. The user has already received final feedback on a research paper 
            but now has follow-up questions.

            Here is the final feedback that was provided by you:
            {final_feedback}

            Here is the summary of the paper:
            {summary}

            Here is the Q&A list from the previous discussion:
            {qa_list}

            Here is the user's follow-up question:
            {query}

            Answer the follow-up question clearly, referring to the final feedback, summary, and Q&A list where relevant.
            If necessary, clarify any points from the final feedback. Keep the response precise and helpful.
            Do not be afraid to also highlight positive AND negative points from your feedback.
            """
        )

        summary = state["summary"]
        qa_list = state["qa_list"]
        final_feedback = state["final_feedback"]
        query = state["questions"][-1]

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"final_feedback": final_feedback, "summary": summary, "qa_list": qa_list, "query": query})
    
        state["response"] = response
        return {"response": response}

    graph_builder.add_node("flag_check", check_generation_feedback)
    graph_builder.add_node("followup_question", follow_up_questions)
    graph_builder.add_node("feedback_or_questions", more_questions_or_not)
    graph_builder.add_node("discuss_paper", discuss_paper)
    graph_builder.add_node("final_feedback_generation", final_feedback)

    graph_builder.add_edge(START, "flag_check")
    graph_builder.add_conditional_edges(
        "flag_check", 
        check_generation_feedback, 
        {"yes": "followup_question", "no": "feedback_or_questions"}
    )
    graph_builder.add_conditional_edges(
        "feedback_or_questions", 
        more_questions_or_not, 
        {"no": "discuss_paper", "yes": "final_feedback_generation"}
    )
    graph_builder.add_edge("followup_question", END)
    graph_builder.add_edge("discuss_paper", END)
    graph_builder.add_edge("final_feedback_generation", END)

    graph = graph_builder.compile()
    return graph
