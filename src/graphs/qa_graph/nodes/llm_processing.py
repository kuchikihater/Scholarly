from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..state import OneLLMState


class LLMProcessingNodes:
    """Nodes for single LLM processing."""

    def single_llm_invoke(self, state: OneLLMState):
        """Generate answer using a single LLM."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages_one_llm"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        document_summary = tool_messages[0].artifact[0].metadata["summary"]

        # Format into prompt
        docs_content = "\n\n".join(
            "\n".join(
                f"{key}: {value}" for key, value in doc.metadata.items() if
                key != "summary") + "\n\n" + doc.page_content for doc in tool_messages[0].artifact
        )

        prompt = PromptTemplate.from_template("""
        You are an assistant for question-answering tasks. See yourself as a professional peer reviewer that gives critical feedback.
        Use the following pieces of retrieved context to answer the question. Use three sentences maximum and keep the answer concise.
        If the question relates to the previous conversation, use the conversation summary to provide the answer. Otherwise, use the retrieved documents and the paper summary.

        For example: if you get questions such as "What is my last question?" -Use conversation summary to provide the answer.
        If you get questions such as "What is the authors of paper?" - Use retrieved documents and the paper summary.

        If you're using the retrieved documents and the question is about the quality of the paper, remember your role as a professional peer reviewer and BE CRITICAL.
        In this case, you can also think a bit before you give an critical answer. DO NOT SAY LET ME THINK, BE PROFESSIONAL

        For example: The question: "Does this paper fulfill the criterion to be published?" - This is about the quality of the paper.
        Whereas questions like: "What is the authors of paper?" - This is a question about facts in the paper, just give the answer you can find in the source.

        Here are the retrieved documents:
        <documents>
        {documents}
        </documents>

        Here is the brief summary of the paper:
        <paper_summary>
        {paper_summary} 
        </paper_summary>

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

        question = state["user_question"]

        chain = prompt | state["llm"] | StrOutputParser()
        summary = state.get("summary", "")
        # Run
        response = chain.invoke(
            {"documents": docs_content, "query": question, "paper_summary": document_summary, "summary": summary})
        return {
            "llms_responses": [
                {
                    "model": getattr(state["llm"], "model_name", "model"),
                    "response": response
                }
            ]
        }
