from functools import wraps

from langchain_core.tools import tool, Tool
from typing import Tuple, List
from langchain_core.documents import Document

from ..state import OverallState


class RetrievalTools:
    """Tools for retrieving document content."""

    def __init__(self, ensemble_retriever, llm):
        """Initialize with an ensemble retriever."""
        self.setup_tools()
        self.ensemble_retriever = ensemble_retriever
        self.llm = llm

    def retrieve(self, query: str) -> Tuple[str, List[Document]]:
        """
        ONLY if in hypothesis mentioned RETRIEVER_TOOL call this.
        DO NOT USE THIS if in hypothesis mentioned GENERAL
        Args:
            query (str): User Question.
        """
        retrieved_docs = self.ensemble_retriever.invoke(query)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def setup_tools(self):
        # Create a wrapper function that doesn't have 'self' conflict
        @tool(response_format="content_and_artifact")
        @wraps(self.retrieve)
        def retrieve_tool(query: str) -> Tuple[str, List[Document]]:
            return self.retrieve(query)

        self.tool = retrieve_tool

    def get_tools(self):
        return self.tool
