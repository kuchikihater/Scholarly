from langgraph.graph import START, END, StateGraph

from src.graphs.qa_graph.state import OneLLMState, OverallState


class SingleLLMSubgraph:
    """Subgraph for processing with a single LLM."""

    def __init__(self, llm_nodes):
        """Initialize with LLM processing nodes."""
        self.llm_nodes = llm_nodes

    def build(self):
        """Build and return the subgraph for single LLM processing."""
        subgraph_builder = StateGraph(OneLLMState, output=OverallState)

        # Add Single LLM Invoke node
        subgraph_builder.add_node("Single LLM Invoke", self.llm_nodes.single_llm_invoke)

        # Add edges
        subgraph_builder.add_edge(START, "Single LLM Invoke")
        subgraph_builder.add_edge("Single LLM Invoke", END)

        return subgraph_builder.compile()