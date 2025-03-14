from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.graphs.feedback_graph.state import State


class FinalFeedbackNode:
    def __init__(self, llm):
        """Initialize with LLM model."""
        self.llm = llm

    def generate_feedback(self, state: State):
        """Generate a final feedback."""
        prompt = PromptTemplate.from_template(
            """
            You are an AI Assistant tasked with providing a recommendation on whether a scientific paper should be published.
            You take on the role of a subreviewer for a peer review of a scientific paper. The user is the main reviewer.

            Your recommendation should be based on:
            1) The provided summary of the paper.
            2) A list of questions and answers related to the paper, representing a conversation about the paper.

            Your response should be structured as follows:

            Recommended: [Your recommendation - either "Accept" or "Reject"]
            Conversation Summary: [Briefly summarize the key points, questions, and concerns raised in the Q&A interaction.]
            Reasoning: [Provide a clear and critical explanation of the reasoning behind your recommendation. 
            Refer to specific aspects of the paper summary and the Q&A list. Highlight both positive and negative points.]
            Improvements Before Publishing: [Provide a concise list of actionable suggestions for improving the paper 
            before publication. Focus on the most important areas for improvement. Use bullet points.
            Consider BOTH:
                a) Specific issues and concerns raised in the Q&A.
                b) General improvements related to the paper's structure, clarity, completeness, and presentation, based on the 
                paper summary.
            ]

            Here is the summary of the paper:
            {summary}

            Here is the Q&A list about the paper (representing a conversation):
            {qa_list}

            Analyze the provided information and produce your recommendation, conversation summary, reasoning, 
            and improvement suggestions.

            Finally, ask the main reviewer (the user) if they have any follow-up questions.
            """

        )

        summary = state["summary"]
        qa_list = state["qa_list"]

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"summary": summary, "qa_list": qa_list})


        return {"final_feedback": response, "response": response, "flag": 1}