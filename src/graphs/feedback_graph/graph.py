from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.graphs.feedback_graph.nodes.follow_up_decision import FollowUpDecisionNode
from src.graphs.feedback_graph.nodes.follow_up_question import FollowUpQuestionNode
from src.graphs.feedback_graph.nodes.pre_feedback_question import PreFeedbackQuestionNode
from src.graphs.feedback_graph.nodes.pre_feedback_decision import PreFeedbackDecisionNode
from src.graphs.feedback_graph.nodes.final_feedback import FinalFeedbackNode

from src.graphs.feedback_graph.state import State


class GraphBuilder:
    """Main graph builder class."""

    def __init__(self, llm):
        """Initialize with necessary models and retrievers."""
        self.llm = llm
        self.memory = MemorySaver()

        # Initialize node classes
        self.follow_up_question_node = FollowUpQuestionNode(llm)
        self.pre_feedback_question_node = PreFeedbackQuestionNode(llm)
        self.follow_up_decision_node = FollowUpDecisionNode()
        self.pre_feedback_decision_node = PreFeedbackDecisionNode(llm)
        self.final_feedback_node = FinalFeedbackNode(llm)


    def build(self):
        """Build the main graph."""
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("answer_follow_up", self.follow_up_question_node.answer_follow_up_question)
        graph_builder.add_node("check_pre_feedback", self.pre_feedback_decision_node.get_summary)
        graph_builder.add_node("answer_pre_feedback_question", self.pre_feedback_question_node.answer_question)
        graph_builder.add_node("generate_final_feedback", self.final_feedback_node.generate_feedback)

        # Add conditional edges
        graph_builder.add_conditional_edges(
            START,
            self.follow_up_decision_node.should_answer_follow_up,
            {"yes": "answer_follow_up", "no": "check_pre_feedback"}
        )
        graph_builder.add_conditional_edges(
            "check_pre_feedback",
            self.pre_feedback_decision_node.should_generate_feedback,
            {"no": "answer_pre_feedback_question", "yes": "generate_final_feedback"}
        )

        # Add edges
        graph_builder.add_edge("answer_follow_up", END)
        graph_builder.add_edge("answer_pre_feedback_question", END)
        graph_builder.add_edge("generate_final_feedback", END)

        # Compile graph
        graph = graph_builder.compile(checkpointer=self.memory)
        
        return graph