from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.graphs.feedback_graph.state import State


class FollowUpQuestionNode:
    def __init__(self, llm):
        """Initialize with LLM model."""
        self.llm = llm

    def answer_follow_up_question(self, state: State):
        """Answer user's follow-up questions after final feedback generation."""
        prompt = PromptTemplate.from_template(
            """
            You are a subreviewer for a peer reviewing of a research paper continuing a scientific peer review discussion. The user has already received final feedback on a research paper 
            but now has follow-up questions.

            Here is the final feedback that was provided by you:
            {final_feedback}

            Here is the summary of the paper:
            {summary}

            Here is the Q&A list from the previous discussion:
            {qa_list}

            Here is the user's follow-up question:
            {query}

            Answer the follow-up question clearly, referring to the final feedback, summary, and Q&A list where relevant.
            If necessary, clarify any points from the final feedback. Keep the response precise and helpful.
            Do not be afraid to also highlight positive AND negative points from your feedback.
            """
        )

        summary = state["summary"]
        qa_list = state["qa_list"]
        final_feedback = state["final_feedback"]
        query = state["questions"][-1]

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke(
            {"final_feedback": final_feedback, "summary": summary, "qa_list": qa_list, "query": query})

        state["response"] = response

        return {"response": response}