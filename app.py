import uuid
from typing import Any

import anthropic
from langgraph.checkpoint.memory import MemorySaver
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.runtime import Runtime
import streamlit as st

import os
import re
import json
from typing_extensions import List, TypedDict, Annotated
from dotenv import load_dotenv
import operator

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, AnyMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from langgraph.graph import START, END, StateGraph, MessagesState, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Send

from pydantic import BaseModel, Field

load_dotenv()


def initialization(file: str):
    class OverallState(MessagesState):
        questions: Annotated[List[HumanMessage], operator.add]
        llms_responses: Annotated[List[dict], operator.add]
        best_responses: Annotated[List[str], operator.add]
        summary: str

    class OneLLMState(MessagesState):
        llm_responses: Annotated[List[AnyMessage], operator.add]
        llm: Any
        best_response: str
        user_question: str

    class OneQuestionResponseState(MessagesState):
        response: AnyMessage
        llm: Any
        user_question: str

    def extract_json_output(response: str):
        json_match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
        json_string = json_match.group(1).strip()
        parsed_json = json.loads(json_string)
        return parsed_json

    def extract_str_output(response: str):
        str_match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
        str_string = str_match.group(1).strip()
        return str_string

    client = anthropic.Anthropic(
        # This is the default and can be omitted
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    llm1 = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    llm2 = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    llm3 = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    llms = [llm1, llm2, llm3]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))
    loader = PyPDFLoader(file_path=file, extract_images=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = loader.load()
    all_splits = text_splitter.split_documents(docs)
    full_document_content = "\n\n".join(doc.page_content for doc in docs)

    DOCUMENT_CONTEXT_PROMPT = """
    <document>
    {doc_content}
    </document>
    """

    CHUNK_CONTEXT_PROMPT = """
    Here is the chunk we want to situate within the whole document
    <chunk>
    {chunk_content}
    </chunk>

    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    """

    # context_create_chain = llm | StrOutputParser()
    #
    # for i, split in enumerate(all_splits):
    #     response = client.messages.create(
    #         model="claude-3-haiku-20240307",
    #         max_tokens=1024,
    #         temperature=0.0,
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "text",
    #                         "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=full_document_content),
    #                         "cache_control": {"type": "ephemeral"}
    #                         # we will make use of prompt caching for the full documents
    #                     },
    #                     {
    #                         "type": "text",
    #                         "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=split),
    #                     }
    #                 ]
    #             }
    #         ],
    #         extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    #     )
    #     all_splits[i].page_content += "/n" + response.content[0].text
    #     print(all_splits[i].page_content)
    db = DocArrayInMemorySearch.from_documents(all_splits, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    memory = MemorySaver()

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """
        If the user has a specific question, you should extract the question and call this function.
        Args:
            query (str): User Question.
        """
        retrieved_docs = retriever.get_relevant_documents(query)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def start_single_llm(state: OneLLMState):
        return state

    def continue_to_generate_llm_invokes(state: OneLLMState):
        return [Send("Single LLM Invoke",
                     {"messages": state["messages"], "response": "", "user_question": state["user_question"],
                      "llm": llm}) for llm in llms]

    def single_llm_invoke(state: OneQuestionResponseState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        prompt = PromptTemplate.from_template("""
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
         Here is retrieved documents:
        <documents>
        {documents}
        </documents>

        And here is the initial query from the user:

        <question>
        {query}
        </question>
        """
                                              )
        question = state["user_question"]

        chain = prompt | state["llm"] | StrOutputParser()
        # Run
        response = chain.invoke({"documents": docs_content, "query": question})
        return {"llm_responses": [response]}

    def choose_best_response_single_llm(state: OneLLMState):
        prompt = PromptTemplate.from_template("""Below are a responses to the user query. Select the best one and return it, without any additional text 
        Here user question:
        {question}
        Here models responses:
        {responses}""")

        chain = prompt | llm
        question = state["user_question"]
        responses = state["llm_responses"]

        response = chain.invoke(
            {
                "question": question,
                "responses": responses,
            }
        )

        state["messages"] = []

        return {"best_response": response.content}

    def return_to_main(state: OneLLMState):
        return {"llms_responses": [{"model": state["llm"].model_name, "best_response": state["best_response"]}]}

    subgraph_builder = StateGraph(OneLLMState, output=OverallState)
    subgraph_builder.add_node("Start Single LLM", start_single_llm)
    subgraph_builder.add_node("Choose Best Response of Single LLM", choose_best_response_single_llm)
    subgraph_builder.add_node("Single LLM Invoke", single_llm_invoke)
    subgraph_builder.add_node("Return To Main Graph", return_to_main)
    subgraph_builder.add_edge(START, "Start Single LLM")
    subgraph_builder.add_conditional_edges("Start Single LLM", continue_to_generate_llm_invokes, ["Single LLM Invoke"])
    subgraph_builder.add_edge("Single LLM Invoke", "Choose Best Response of Single LLM")
    subgraph_builder.add_edge("Choose Best Response of Single LLM", "Return To Main Graph")
    subgraph_builder.add_edge("Return To Main Graph", END)

    def generate_feedback_or_not(state: OverallState):
        prompt = PromptTemplate.from_template(
            """
            Your task is determine, if user wants to generate final feeedback or not
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
            return "yes"
        if "no" == response_json["feedback"]:
            return "no"

    def retrieve_or_not(state: OverallState):
        user_message = {"role": "user", "content": state["questions"][-1]}
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke([user_message])
        if len(response.tool_calls) == 0:
            return {"messages": [response], "best_responses": [response.content]}
        else:
            return {"messages": [response]}

    tools = ToolNode([retrieve])

    def evaluate_documents(state: OverallState):
        print(state["questions"])
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
        if filtered_response["relevant"] == "yes":
            return "Start Generate LLMs"
        elif filtered_response["relevant"] == "no":
            del state["messages"][-1]
            return "Rewrite User Question"

    def rewrite_user_question(state: OverallState):
        prompt = PromptTemplate.from_template(
            """
            You are an advanced language model tasked with improving user queries to enhance document retrieval and overall conversation quality. Your goal is to analyze the initial query and conversation history, understand the underlying semantic intent, and formulate an improved question.

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
            After your analysis, provide only the improved question wraped in <output> without any additional explanation or text.

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
        # messages = state["summary"]
        messages = ""

        response = chain.invoke(
            {
                "messages": messages,
                "query": question,
            }
        )

        del state["questions"][-1]

        return {"messages": HumanMessage(content=extract_str_output(response)),
                "questions": extract_str_output(response)}

    def start_generate_llms(state: OverallState):
        print(len(state["questions"]))
        return state

    def continue_to_generate_llms(state: OverallState):
        return [Send("Single LLM Process Start",
                     {"messages": state["messages"], "response": "", "user_question": state["questions"][-1],
                      "llm": llm}) for llm in llms]

    def best_response(state: OverallState):
        print(len(state["questions"]))
        prompt = PromptTemplate.from_template("""Below are a responses to the user query. Select the best one and return it, without any additional text 
        Here user question:
        {question}
        Here models responses:
        {responses}""")

        chain = prompt | llm
        question = state["questions"][-1]
        responses = state["llms_responses"]

        response = chain.invoke(
            {
                "question": question,
                "responses": responses,
            }
        )

        state["messages"] = []

        return {"best_responses": [response.content]}

    def generate_feedback(state: OverallState):
        prompt = PromptTemplate.from_template(
            """
            Your task is to generate summarization of conversation
            Here is the user question:
            Here is the history of the conversation:
            <conversation_history>
            {messages}
            </conversation_history>
            """
        )

        chain = prompt | llm
        messages = state["messages"][-3]

        response = chain.invoke(
            {
                "messages": messages,
            }
        )

        return {"messages": [response]}

    graph_builder = StateGraph(OverallState)
    graph_builder.add_node("Direct Answer or Retrieve", retrieve_or_not)
    graph_builder.add_node("Retrieve Documents", tools)
    graph_builder.add_node("Single LLM Process Start", subgraph_builder.compile())
    graph_builder.add_node("Start Generate LLMs", start_generate_llms)
    graph_builder.add_node("Rewrite User Question", rewrite_user_question)
    graph_builder.add_node("Generate Feedback", generate_feedback)
    graph_builder.add_node("Choose Best Response", best_response)

    graph_builder.add_conditional_edges(
        START,
        generate_feedback_or_not,
        {"yes": "Generate Feedback", "no": "Direct Answer or Retrieve"},
    )

    graph_builder.add_conditional_edges(
        "Direct Answer or Retrieve",
        tools_condition,
        {END: END, "tools": "Retrieve Documents"},
    )
    graph_builder.add_conditional_edges(
        "Retrieve Documents",
        evaluate_documents,
        {"Start Generate LLMs": "Start Generate LLMs", "Rewrite User Question": "Rewrite User Question"},
    )
    graph_builder.add_edge("Rewrite User Question", "Direct Answer or Retrieve")
    graph_builder.add_conditional_edges("Start Generate LLMs", continue_to_generate_llms, ["Single LLM Process Start"])
    graph_builder.add_edge("Single LLM Process Start", "Choose Best Response")
    graph_builder.add_edge("Choose Best Response", END)
    graph_builder.add_edge("Generate Feedback", END)

    graph = graph_builder.compile(checkpointer=memory)

    return graph


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

st.title("Hey there! I'm Scholarly. Ready to review your paper and give you feedback. Let’s get started!")
uploaded_file = st.file_uploader('Upload your paper in .pdf format', type="pdf")

if uploaded_file is not None:
    temp_file_path = "/tmp/uploaded_paper.pdf"
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            graph = initialization(temp_file_path)
        except Exception as e:
            st.error(f"Error initializing the graph: {e}")
            graph = None

        if graph:
            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            prompt = st.chat_input("Hi, what do you want to ask?")
            st.write("")

            memory = {
                questions: [],
                answers: []
            }

            if prompt:
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
                try:
                    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
                    stream = graph.stream({"questions": [prompt]}, stream_mode="values",
                                          config=config)
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        response = ""
                        for msg in stream:
                            if msg["best_responses"]:
                                response += msg["best_responses"][-1]
                                print(len(msg["best_responses"]))
                                response_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Graph initialization failed. Please check your input file.")
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
else:
    st.info("Please upload a valid PDF file.")
