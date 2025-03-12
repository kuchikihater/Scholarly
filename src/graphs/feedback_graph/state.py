import operator
from typing import List, TypedDict, Annotated

from langchain_core.messages import HumanMessage


class State(TypedDict):
    questions: Annotated[List[HumanMessage], operator.add]
    summary: str
    qa_list: list
    response: str
    final_feedback: str
    flag: int