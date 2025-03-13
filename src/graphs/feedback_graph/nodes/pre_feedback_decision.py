from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.graphs.feedback_graph.state import State

from src.utils.extractors import extract_json_output


class PreFeedbackDecisionNode:
    def __init__(self, llm):
        """Initialize with LLM model."""
        self.llm = llm

    def should_generate_feedback(self, state: State):
        "Check if generate direct feedback or answer questions first."
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

    @staticmethod
    def get_summary(state: State):
        """Retrieve paper summary from the session state."""
        return {"summary": state["summary"]}