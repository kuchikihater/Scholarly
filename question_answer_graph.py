import os
import re
import json

from typing_extensions import List, TypedDict, Annotated, Any
from dotenv import load_dotenv
import operator

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_anthropic import ChatAnthropic

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Send

load_dotenv()


def initialization(file: str):
    ###Extract output from tags in string format
    print(1)
    def extract_json_output(response: str):
        json_match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
        json_string = json_match.group(1).strip()
        parsed_json = json.loads(json_string)
        return parsed_json

    ###Extract output from tags in string format
    def extract_str_output(response: str):
        str_match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
        str_string = str_match.group(1).strip()
        return str_string

    class OverallState(MessagesState):
        questions: Annotated[List[HumanMessage], operator.add]
        llms_responses: Annotated[List[Any], operator.add]
        end_responses: Annotated[List[str], operator.add]
        hypothesis: str
        retrieved_documents: List[Document]
        attempt: int
        summary: str

    class OneLLMState(TypedDict):
        messages_one_llm: Annotated[List[AnyMessage], operator.add]
        llm_responses: Annotated[List[AnyMessage], operator.add]
        llm: Any
        user_question: str
        summary: str


    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    llm1 = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    llm2 = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    llm3 = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    # llm3 = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=os.getenv("ANTHROPIC_API_KEY"))
    llms = [llm1, llm2, llm3]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))
    loader = PyPDFLoader(file_path=file, extract_images=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=0)
    docs = loader.load()
    all_splits = text_splitter.split_documents(docs)
    full_document_content = "\n\n".join(doc.page_content for doc in docs)
    # Define prompt
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Write a concise summary of the following:\\n\\n{context}")]
    )

    # Instantiate chain
    chain = create_stuff_documents_chain(llm, prompt)

    # Invoke chain
    result = chain.invoke({"context": docs})

    for doc in all_splits:
        doc.metadata['summary'] = result

    # initialize the bm25 retriever and faiss retriever
    bm25_retriever = BM25Retriever.from_documents(
        all_splits
    )
    bm25_retriever.k = 2
    embedding = OpenAIEmbeddings()
    faiss_vectorstore = FAISS.from_documents(
        all_splits, embedding
    )
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )

    memory = MemorySaver()

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """
        ONLY if in hypothesis mentioned RETRIEVER_TOOL call this.
        DO NOT USE THIS if in hypothesis mentioned GENERAL
        Args:
            query (str): User Question.
        """

        retrieved_docs = ensemble_retriever.invoke(query)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def single_llm_invoke(state: OneLLMState):
        """Generate answer."""
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
        In this case, you can also think a bit before you give an critical answer.
        
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

    subgraph_builder = StateGraph(OneLLMState, output=OverallState)
    subgraph_builder.add_node("Single LLM Invoke", single_llm_invoke)
    subgraph_builder.add_edge(START, "Single LLM Invoke")
    subgraph_builder.add_edge("Single LLM Invoke", END)

    def generate_feedback_or_not(state: OverallState):
        prompt = PromptTemplate.from_template(
            """
            Your task is to determine, if the user wants to generate final feedback or not
            Here is the user question:
            <question>
            {query}
            </question>
            After your assessment, provide your final decision in JSON format. The JSON must contain a single key "feedback" with a value of either "yes" or "no". For example:

            {{
              "feedback": "yes"
            }}

            or

            {{
              "feedback": "no"
            }}

            Wrap your answer in <output> tag.
            """
        )

        chain = prompt | llm | StrOutputParser()
        question = state["questions"][-1]

        response = chain.invoke({"query": question})
        response_json = extract_json_output(response)

        if "yes" == response_json["feedback"]:
            return "no"
        if "no" == response_json["feedback"]:
            return "no"

    def make_hypothesis(state: OverallState):
        prompt = PromptTemplate.from_template(
            """
            You are an AI assistant that specializes in analyzing user questions about a scientific paper and tells whether or not the context of the paper is needed to answer the question. 
            Your job is to create a reasoned hypothesis about whether or not to use RETRIEVER_TOOL.

            RETRIEVER_TOOL: It is the tool, that retrieves parts of paper

            Please follow these steps to create your hypothesis:

            1. Carefully read and analyze the user question.
            2. Decide, if user question is related to scientific paper or not.
            3. When the user question is related to scientific paper then return then return RETRIEVER_TOOL
            4. When the user question is about the previous question, history of conversation or general message, then return GENERAL

            At the end ALWAYS add the user question to response
            Example of answer: RETRIEVER_TOOL, USER_QUESTION: About what this paper?

            Here is the user question:
            <question>
            {query}
            </question>
            """
        )

        llm = ChatOpenAI(model="gpt-4o")

        chain = prompt | llm | StrOutputParser()
        question = state["questions"][-1]

        response = chain.invoke({"query": question})

        return {"hypothesis": response}

    def retrieve_or_not(state: OverallState):
        if "RETRIEVER_TOOL" in state["hypothesis"]:
            llm_with_tools = llm.bind_tools([retrieve])
        else:
            llm_with_tools = llm

        response = llm_with_tools.invoke([state["hypothesis"]])

        if len(response.tool_calls) == 0:
            return {"messages": [response], "end_responses": [response.content]}
        else:
            if "attempt" in state:
                return {"messages": [response], "attempt": state["attempt"]}
            return {"messages": [response], "attempt": 1}

    tools = ToolNode([retrieve])

    def evaluate_documents(state: OverallState, max_retries=2):
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

            Wrap your answer in <output> tag.
            """
        )

        parser = StrOutputParser()

        chain = prompt | llm | parser

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
                          "llm": llm}) for llm in llms]
        elif filtered_response["relevant"] == "no":
            del state["messages"][-1]
            return "Rewrite User Question"

    def rewrite_user_question(state: OverallState):
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
        chain = prompt | llm | parser

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

    def start_generate_llms(state: OverallState):
        return {}

    def continue_to_generate_llms(state: OverallState):
        summary = state.get("summary", "")
        return [Send("Single LLM Process Start",
                     {"messages_one_llm": state["messages"], "response": "", "user_question": state["questions"][-1],
                      "llm": llm}) for llm in llms]

    def end_response(state: OverallState):
        """
        This function combines the last three LLM responses into a single formatted string,
        saves it in `state["end_responses"]`, and clears the `state["messages"]`.
        """
        # Retrieve the last three responses from llm_responses
        responses = state["llms_responses"][-3:]

        # Create a combined response string
        combined_response = "\n ".join(
            [f"{i + 1}) Answer of {response['model']} : {response['response']}" for i, response in enumerate(responses)]
        )

        # Save the combined response in state["end_responses"]
        state["end_responses"] = [combined_response]

        # Clear the messages in state
        state["messages"] = [RemoveMessage(id=m.id) for m in state["messages"]]

        return {"end_responses": state["end_responses"], "messages": state["messages"]}

    def generate_feedback(state: OverallState):
        """
        Generate Feedback
        :param state:
        :return:
        """

    def generate_summary(state: OverallState):
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

        chain = prompt | llm | StrOutputParser()
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

    def generate_simple_response(state: OverallState):
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

        chain = prompt | llm | StrOutputParser()
        summary = state.get("summary", "")
        # Run
        response = chain.invoke({"query": question, "summary": summary})
        return {"messages": [response], "end_responses": [response]}

    graph_builder = StateGraph(OverallState)
    graph_builder.add_node("Make Hypothesis", make_hypothesis)
    graph_builder.add_node("Direct Answer or Retrieve", retrieve_or_not)
    graph_builder.add_node("Retrieve Documents", tools)
    graph_builder.add_node("Single LLM Process Start", subgraph_builder.compile())
    graph_builder.add_node("Rewrite User Question", rewrite_user_question)
    graph_builder.add_node("Generate Feedback", generate_feedback)
    graph_builder.add_node("Give End Response", end_response)
    graph_builder.add_node("Give Simple Response", generate_simple_response)
    graph_builder.add_node("Generate Summary", generate_summary)



    graph_builder.add_conditional_edges(
        START,
        generate_feedback_or_not,
        {"yes": "Generate Feedback", "no": "Make Hypothesis"},
    )

    graph_builder.add_edge("Make Hypothesis", "Direct Answer or Retrieve")

    graph_builder.add_conditional_edges(
        "Direct Answer or Retrieve",
        tools_condition,
        {END: "Give Simple Response", "tools": "Retrieve Documents"},
    )

    graph_builder.add_conditional_edges(
        "Retrieve Documents",
        evaluate_documents,
        {"Start Generate LLMs": "Single LLM Process Start", "Rewrite User Question": "Rewrite User Question"},
    )
    graph_builder.add_edge("Rewrite User Question", "Direct Answer or Retrieve")
    graph_builder.add_edge("Single LLM Process Start", "Give End Response")
    graph_builder.add_edge("Give End Response", "Generate Summary")
    graph_builder.add_edge("Generate Summary", END)
    graph_builder.add_edge("Generate Feedback", END)
    graph_builder.add_edge("Give Simple Response", END)

    graph = graph_builder.compile(checkpointer=memory)

    return graph, result