from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.runtime import Runtime
import streamlit as st

from dotenv import load_dotenv
import os
from typing_extensions import List, TypedDict

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


load_dotenv()


def initialization(file: str):
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = Chroma(embedding_function=embeddings)
    loader = PyPDFLoader(file_path=file, extract_images=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    docs = loader.load()
    all_splits = text_splitter.split_documents(docs)
    _ = vector_store.add_documents(documents=all_splits)

    graph_builder = StateGraph(MessagesState)

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """
        A tool for finding relevant information in uploaded documents based on a user's query.

        Use this tool when you need to find specific information or context from documents to answer a user's question.
        The tool searches the vector store and returns the most relevant pieces of text.

        Output Format:
            - Content: Compiled text from the retrieved documents.
            - Artifact: A list of documents or fragments with additional metadata.
        """
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def query_or_respond(state: MessagesState):
        llm_with_tools = llm.bind_tools([retrieve])
        # system_msg = """
        # You are an intelligent assistant that answers user questions by using available tools when necessary.
        # You have access to the following tools:
        # - **retrieve**: A tool for searching relevant information in the loaded documents based on the user's query.
        # "Use this tool when you need to obtain specific information from the documents to answer the user's question.
        # Rephrase or elaborate on the query if necessary to make it clearer and more effective for retrieval.
        # When you want to use a tool, follow this format:
        # Action: [tool name]
        # Action Input: [input for the tool]
        # """
        prompt = state["messages"]
        response = llm_with_tools.invoke(prompt)
        return {"messages": [response]}

    tools = ToolNode([retrieve])

    def generate(state: MessagesState):
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
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
               or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()

    return graph


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

            if prompt:
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
                try:
                    stream = graph.stream({"messages": [{"role": "user", "content": prompt}]}, stream_mode="messages")
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        response = ""
                        for msg, metadata in stream:
                            if msg.content and metadata.get("langgraph_node") in ["generate", "query_or_respond"]:
                                response += msg.content
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

