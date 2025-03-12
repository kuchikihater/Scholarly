from src.graphs.feedback_graph.state import State


class FollowUpDecisionNode:
    def should_answer_follow_up(self, state: State):
        """Determines if the user is asking a follow-up question after feedback generation.

        Args:
            state (State): The current graph state.

        Returns:
            str: "yes" if the user is asking a follow-up question (feedback has been generated),
                 "no" otherwise (feedback has not yet been generated).
        """
        return "yes" if state.get("flag", 0) == 1 else "no"