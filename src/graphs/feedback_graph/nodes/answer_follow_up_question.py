from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.graphs.config import OPENAI_MODEL_GPT4O
from src.graphs.feedback_graph.state import State

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
