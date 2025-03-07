from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.graphs.config import OPENAI_MODEL_GPT4O
from src.graphs.feedback_graph.state import State

llm = ChatOpenAI(model=OPENAI_MODEL_GPT4O)


class CheckGenerationFeedback:
    def check_generation_feedback(self, state: State):
        "Check Generation Feedback"
        response = "yes" if state.get("flag", 0) == 1 else "no"
        return response

