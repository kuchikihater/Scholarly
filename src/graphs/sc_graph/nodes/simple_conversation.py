from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.graphs.sc_graph.state import State


class SimpleConversationNodes:
    """Nodes for initial simple conversation."""

    def __init__(self, llm):
        """Initialize with LLM model."""
        self.llm = llm

    def chatbot(self, state: State):
        """Generate a simple conversation."""
        prompt = PromptTemplate.from_template(
            """
            You are an AI Assistant.
            
            ADD after each your response reminder to the user that you can answer on question about paper if he would upload that
    
            Here is the initial query from the user:
            <question>
            {query} 
            </question>
            """
        )

        question = state["messages"][-1]

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": question})

        return {"messages": [response]}