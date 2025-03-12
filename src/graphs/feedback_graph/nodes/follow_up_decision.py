from src.graphs.feedback_graph.state import State


class FollowUpDecisionNode:
    def should_answer_follow_up(self, state: State):
        """
        Check if answer follow-up question or generate feedback.
        
        If flag is set to 1 (meaning true), it means feedback has already been generated and the user question is follow-up question.
        
        Otherwise, we decide whether to generate direct feedback or discuss paper first.
        """
        response = "yes" if state.get("flag", 0) == 1 else "no"
        return response