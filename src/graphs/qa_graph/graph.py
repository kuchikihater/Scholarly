from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from .state import OverallState
from .nodes.hypothesis import HypothesisNodes
from .nodes.retrieval import RetrievalNodes
from .nodes.response import ResponseNodes
from .nodes.tools import RetrievalTools
from .nodes.subgraph import SingleLLMSubgraph
from .nodes.llm_processing import LLMProcessingNodes


class GraphBuilder:
    """Main graph builder class."""

    def __init__(self, llm, llm1, llm2, llm3, ensemble_retriever):
        """Initialize with necessary models and retrievers."""
        self.llm = llm
        self.llm1 = llm1
        self.llm2 = llm2
        self.llm3 = llm3
        self.llms = [llm1, llm2, llm3]
        self.ensemble_retriever = ensemble_retriever
        self.memory = MemorySaver()

        # Initialize node classes
        self.hypothesis_nodes = HypothesisNodes(llm)
        self.response_nodes = ResponseNodes(llm)
        self.llm_processing_nodes = LLMProcessingNodes()
        self.retriever_tools = RetrievalTools(ensemble_retriever, llm)
        self.retrieval_nodes = RetrievalNodes(llm, self.llms, self.retriever_tools.get_tools())

        # Build subgraph
        self.single_llm_subgraph = SingleLLMSubgraph(self.llm_processing_nodes).build()

    def build(self):
        """Build the main graph."""
        graph_builder = StateGraph(OverallState)

        # Add nodes
        graph_builder.add_node("Make Hypothesis", self.hypothesis_nodes.make_hypothesis)
        graph_builder.add_node("Direct Answer or Retrieve", self.retrieval_nodes.retrieve_or_not)
        graph_builder.add_node("Retrieve Documents", ToolNode([self.retriever_tools.get_tools()]))
        graph_builder.add_node("Single LLM Process Start", self.single_llm_subgraph)
        graph_builder.add_node("Rewrite User Question", self.retrieval_nodes.rewrite_user_question)
        graph_builder.add_node("Give End Response", self.response_nodes.end_response)
        graph_builder.add_node("Give Simple Response", self.response_nodes.generate_simple_response)
        graph_builder.add_node("Generate Summary", self.response_nodes.generate_summary)

        # Add conditional edges
        graph_builder.add_edge(START,"Make Hypothesis")

        graph_builder.add_edge("Make Hypothesis", "Direct Answer or Retrieve")

        graph_builder.add_conditional_edges(
            "Direct Answer or Retrieve",
            tools_condition,
            {END: "Give Simple Response", "tools": "Retrieve Documents"},
        )

        graph_builder.add_conditional_edges(
            "Retrieve Documents",
            self.retrieval_nodes.evaluate_documents,
            {"Single LLM Process Start": "Single LLM Process Start", "Rewrite User Question": "Rewrite User Question"},
        )

        graph_builder.add_edge("Rewrite User Question", "Direct Answer or Retrieve")
        graph_builder.add_edge("Single LLM Process Start", "Give End Response")
        graph_builder.add_edge("Give End Response", "Generate Summary")
        graph_builder.add_edge("Generate Summary", END)
        graph_builder.add_edge("Give Simple Response", END)

        # Compile graph
        graph = graph_builder.compile(checkpointer=self.memory)

        return graph
