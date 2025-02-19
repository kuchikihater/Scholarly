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
            You are an AI Assistant tasked with providing a recommendation on whether a scientific paper should be published. 
            Your recommendation should be based on:

            1) The provided summary of the paper.
            2) A list of questions and answers related to the paper, representing a conversation about the paper.

            Your response should be structured as follows:

            **Recommended:** [Your recommendation - either "Accept" or "Reject"]
            **Conversation Summary:** [Briefly summarize the key points, questions, and concerns raised in the Q&A interaction.]
            **Reasoning:** [Provide a clear and critical explanation of the reasoning behind your recommendation. 
            Refer to specific aspects of the paper summary and the Q&A list. Highlight both positive and negative points.]

            [If and only if the recommendation is "Accept", include the following section:]
            **Improvements Before Publishing:** [Provide a concise list of actionable suggestions for improving the paper 
            before publication. Focus on the most important areas for improvement. Use bullet points.
            Consider BOTH:
                a) Specific issues and concerns raised in the Q&A.
                b) General improvements related to the paper's structure, clarity, completeness, and presentation, based on the 
                paper summary.
            ]

            
            Here is the summary of the paper:
            {summary}
            
            Here is the Q&A list about the paper (representing a conversation)::
            {qa_list}
            
            Analyze the provided information and produce your recommendation, conversation summary, reasoning, and (if applicable) 
            improvement suggestions.
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
