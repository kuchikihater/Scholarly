from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.graphs.config import OPENAI_MODEL_GPT4O
from src.graphs.feedback_graph.state import State

from ...utils.helpers import extract_json_output

llm = ChatOpenAI(model=OPENAI_MODEL_GPT4O)


class CheckPaperDiscussion:
    def __init__(self, llm):
        self.llm = llm

    def generate_feedback_or_not(self, state: State):
        "Check if generate feedback or answer on simple question"
        prompt = PromptTemplate.from_template(
            """
            Your task is to determine if the user wants to get final decision or feedback about acceptance of paper or ask
            further questions about the paper.

            User input:
            {query}

            If the user wants you to make final decision or feedback, return:
            {{ "feedback": "yes" }}

            If the user question is something else, return:
            {{ "feedback": "no"}}

            Wrap your answer in <output> tags.
            """
        )

        query = state["questions"][-1]
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query})
        response_json = extract_json_output(response)
        response = "yes" if response_json["feedback"] == "yes" else "no"
        return response

