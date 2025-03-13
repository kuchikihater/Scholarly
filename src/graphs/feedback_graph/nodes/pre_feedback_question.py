from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.graphs.feedback_graph.state import State


class PreFeedbackQuestionNode:
    def __init__(self, llm):
        """Initialize with LLM model."""
        self.llm = llm

    def answer_question(self, state: State):
        """Answer user's questions before generating a final feedback."""
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

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"summary": summary, "qa_list": qa_list, "query": query})

        state["response"] = response

        return {"response": response}