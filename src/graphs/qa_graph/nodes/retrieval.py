from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langgraph.types import Send

from src.graphs.qa_graph.state import OverallState
from src.utils.extractors import extract_json_output, extract_str_output


class RetrievalNodes:
    """Nodes for document retrieval and evaluation."""

    def __init__(self, llm, llms, tools):
        """Initialize with LLM models."""
        self.llm = llm
        self.llms = llms
        self.tools = tools

    def retrieve_or_not(self, state: OverallState):
        if "RETRIEVER_TOOL" in state["hypothesis"]:
            # Pass the tool directly, not the bound method
            llm_with_tools = self.llm.bind_tools([self.tools])
        else:
            llm_with_tools = self.llm

        response = llm_with_tools.invoke([state["hypothesis"]])

        if len(response.tool_calls) == 0:
            return {"messages": [response], "end_responses": [response.content]}
        else:
            if "attempt" in state:
                return {"messages": [response], "attempt": state["attempt"]}
            return {"messages": [response], "attempt": 1}

    def evaluate_documents(self, state: OverallState, max_retries=2):
        """Evaluate if retrieved documents are relevant to the user's question."""
        prompt = PromptTemplate.from_template(
            """
            You are an expert document assessor tasked with determining the relevance of a retrieved document to a user's question. Your goal is to provide an accurate relevance assessment based on both keyword matches and semantic understanding.

            First, carefully review the following information:

            1. Retrieved Documents:
            <retrieved documents>
            {documents}
            </retrieved documents>

            2. Here is the user question:
            <question>
            {query}
            </question>

            Your task is to determine whether the retrieved document is relevant to the user's question. Follow these steps:

            1. Analyze the document and question for keyword matches and semantic relevance.
            2. Consider any information in the document that could be helpful in answering the user's question, even if it's not a direct match.
            3. Make a decision on relevance, erring on the side of relevance if there's any doubt.
            4. Provide your assessment in the specified JSON format.

            Before making your final decision, wrap your thought process in <relevance_assessment> tags:

            <relevance_assessment>
            1. Quote any relevant parts of the document and the question, highlighting keyword matches.
            2. Describe any semantic connections or relevant information found in the document.
            3. List arguments for considering the document relevant.
            4. List arguments for considering the document not relevant.
            5. Explain your reasoning for your final decision on relevance.
            </relevance_assessment>

            After your assessment, provide your final decision in JSON format. The JSON must contain a single key "relevant" with a value of either "yes" or "no". For example:

            {{
              "relevant": "yes"
            }}

            or

            {{
              "relevant": "no"
            }}

            Remember, it's important to consider both direct keyword matches and broader semantic relevance. If the document contains any information that could be helpful in addressing the user's question, even indirectly, it should be considered relevant.

            Wrap your ONLY last answer in <output> tag.
            """
        )

        parser = StrOutputParser()
        chain = prompt | self.llm | parser

        question = state["questions"][-1]
        documents = state["messages"][-1].content
        response = chain.invoke(
            {
                "documents": documents,
                "query": question,
            }
        )

        filtered_response = extract_json_output(response)
        if filtered_response["relevant"] == "yes" or state["attempt"] >= 2:
            return [Send("Single LLM Process Start",
                         {"messages_one_llm": state["messages"], "response": "",
                          "user_question": state["questions"][-1],
                          "llm": llm}) for llm in self.llms]
        elif filtered_response["relevant"] == "no":
            del state["messages"][-1]
            return "Rewrite User Question"

    def rewrite_user_question(self, state: OverallState):
        """Rewrite the user question to improve retrieval effectiveness."""
        prompt = PromptTemplate.from_template(
            """
            You are an advanced language model tasked with improving user queries to enhance document retrieval and overall conversation quality. Your goal is to analyze the initial query and conversation history, understand the underlying semantic intent, and formulate an improved question.

            DO NOT REWRITE QUESTION RELATED TO HISTORY OF CONVERSATION. For example: What is my previous question? JUST GIVE BACK THE QUESTION AS HOW IT WAS

            Here is the history of the conversation:

            <conversation_history>
            {messages}
            </conversation_history>

            And here is the initial query from the user:

            <initial_query>
            {query}
            </initial_query>

            Please follow these steps to formulate an improved question:

            1. Analyze the conversation history and initial query.
            2. Identify the underlying semantic intent or meaning behind the user's question.
            3. Consider any additional context or information provided in the conversation history that could help clarify or refine the query.
            4. Formulate an improved question that:
               - Captures the core intent of the original query
               - Incorporates relevant context from the conversation history
               - Is more precise, clear, and likely to yield better document retrieval results
               - Maintains the original topic and purpose of the query

            Before providing the final improved question, please wrap your thought process inside <query_improvement_process> tags. This will help ensure a thorough interpretation of the query and conversation context. In this process:

            1. Identify and list key topics/themes from the conversation history.
            2. Quote relevant parts of the conversation history that provide context for the query.
            3. Break down the initial query into its core components.
            4. Consider how the conversation history might influence or refine each component of the query.
            5. Explain how you arrived at the improved question based on this analysis.

            Output Format:
            After your analysis, provide only the improved question wrapped in <output> without any additional explanation or text.

            Example output structure:

            <query_improvement_process>
            [Your detailed analysis of the conversation history and initial query, following the steps outlined above]
            </query_improvement_process>

            <output>
            [Improved question goes here, without any tags or additional text]
            </output>

            Please proceed with your query improvement process and improved question formulation.      
            """
        )

        parser = StrOutputParser()
        chain = prompt | self.llm | parser

        question = state["questions"][-1]
        messages = state.get("summary", "")

        response = chain.invoke(
            {
                "messages": messages,
                "query": question,
            }
        )

        del state["questions"][-1]

        return {"messages": [HumanMessage(content=extract_str_output(response))],
                "questions": [extract_str_output(response)], "attempt": state["attempt"] + 1}
