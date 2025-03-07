from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from ..state import OverallState
from ...utils.helpers import extract_json_output


class HypothesisNodes:
    """Nodes for hypothesis generation and decision-making."""

    def __init__(self, llm):
        """Initialize with LLM models."""
        self.llm = llm

    def make_hypothesis(self, state: OverallState):
        """Generate hypothesis about whether to use retrieval tool."""
        prompt = PromptTemplate.from_template(
            """
            You are an AI assistant that specializes in analyzing user questions about a scientific paper and tells whether or not the context of the paper is needed to answer the question. 
            Your job is to create a reasoned hypothesis about whether or not to use RETRIEVER_TOOL.

            RETRIEVER_TOOL: It is the tool, that retrieves parts of paper

            Please follow these steps to create your hypothesis:

            1. Carefully read and analyze the user question.
            2. Decide, if user question is related to scientific paper or not.
            3. When the user question is related to scientific paper then return then return RETRIEVER_TOOL
            4. When the user question is about the previous question, history of conversation or general message, then return GENERAL

            At the end ALWAYS add the user question to response
            Example of answer: RETRIEVER_TOOL, USER_QUESTION: About what this paper?

            Here is the user question:
            <question>
            {query}
            </question>
            """
        )

        chain = prompt | self.llm | StrOutputParser()
        question = state["questions"][-1]

        response = chain.invoke({"query": question})

        return {"hypothesis": response}
