from langgraph.graph import StateGraph, START, END

from src.graphs.sc_graph.nodes.simple_conversation import SimpleConversationNodes
from src.graphs.sc_graph.state import State


class GraphBuilder:
    """Main graph builder class."""

    def __init__(self, llm):
        """Initialize with necessary models and retrievers."""
        self.llm = llm

        # Initialize node classes
        self.simple_conversation_nodes = SimpleConversationNodes(llm)


    def build(self):
        """Build the main graph."""
        graph_builder = StateGraph(State)
    
        graph_builder.add_node("chatbot", self.simple_conversation_nodes.chatbot)

        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        graph = graph_builder.compile()

        return graph