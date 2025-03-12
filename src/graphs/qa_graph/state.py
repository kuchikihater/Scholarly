import operator
from typing import List, TypedDict, Any, Annotated

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import MessagesState


class OverallState(MessagesState):
    questions: Annotated[List[HumanMessage], operator.add]
    llms_responses: Annotated[List[Any], operator.add]
    end_responses: Annotated[List[str], operator.add]
    hypothesis: str
    retrieved_documents: List[Document]
    attempt: int
    summary: str


class OneLLMState(TypedDict):
    messages_one_llm: Annotated[List[AnyMessage], operator.add]
    llm_responses: Annotated[List[AnyMessage], operator.add]
    llm: Any
    user_question: str
    summary: str

