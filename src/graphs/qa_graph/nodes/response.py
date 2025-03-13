from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import RemoveMessage

from ..state import OverallState


class ResponseNodes:
    """Nodes for generating responses and summaries."""

    def __init__(self, llm):
        """Initialize with LLM model."""
        self.llm = llm

    def end_response(self, state: OverallState):
        """Combine the last three LLM responses into a single formatted response."""
        # Retrieve the last three responses from llm_responses
        responses = state["llms_responses"][-3:]

        # Create a combined response string
        combined_response = "\n ".join(
            [f"{i + 1}) Answer of Reviewer {i + 1} : {response['response']}" for i, response in enumerate(responses)]
        )

        # Save the combined response in state["end_responses"]
        state["end_responses"] = [combined_response]

        # Clear the messages in state
        state["messages"] = [RemoveMessage(id=m.id) for m in state["messages"]]

        return {"end_responses": state["end_responses"], "messages": state["messages"]}

    def generate_summary(self, state: OverallState):
        """Generate a summary of the conversation so far."""
        prompt = PromptTemplate.from_template(
            """
            Your task is to summarize the whole conversation so far, including the user quesiton, the
            ai response and the previous summary if available. The summarization should also serve as memory such that when the user asks about information mentioned above, it can be used to answer such questions. The followings are the information you need:

            1. Here is the user question:
            <question>
            {question}
            </question>

            2. Here is the ai response:
            <response>
            {answer}
            </response>

            3. Here is the previous summary:
            <summary>
            {summary}
            </summary>

            If the summary is empty, just ignore it. Otherwise, pass the questions and responses that are in summary already into the new summary as well. Combine the information you have and summarize it into a maximal-200-word summary.
            """
        )

        chain = prompt | self.llm | StrOutputParser()
        question = state["questions"][-1]
        answer = state["end_responses"][-1]
        summary = state.get("summary", "")

        response = chain.invoke(
            {
                "question": question,
                "answer": answer,
                "summary": summary
            }
        )

        # Delete all but the 2 most recent messages
        delete_questions = [question]
        delete_end_responses = [answer]
        state["questions"].clear()
        state["end_responses"].clear()
        return {"summary": response, "questions": delete_questions, "end_responses": delete_end_responses}

    def generate_simple_response(self, state: OverallState):
        """Generate a simple response for general questions."""
        prompt = PromptTemplate.from_template(
            """
            You are an AI Assistant that helps with question and answering about conversation

            Here is the initial query from the user:
            <question>
            {query} 
            </question>

            Here is the summary of the previous conversation:
            <conversation_summary>
            {summary} 
            </conversation_summary>
            """
        )

        question = state["questions"][-1]

        chain = prompt | self.llm | StrOutputParser()
        summary = state.get("summary", "")
        # Run
        response = chain.invoke({"query": question, "summary": summary})
        return {"messages": [response], "end_responses": [response]}
