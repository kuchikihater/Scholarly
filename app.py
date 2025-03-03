import uuid
import streamlit as st
from dotenv import load_dotenv

from question_answer_graph import initialization as qa_initialization
from simple_conversation import initialization as simple_conversation_initialization
from final_feedback_conversation import initialization as final_feedback_conversation_initialization

from streamlit_float import *

load_dotenv()
st.set_page_config(layout="wide")

float_init(theme=True, include_unstable_primary=False)


def chat_content():
    if st.session_state.get('prebuilt_question', ""):
        user_input = st.session_state.prebuilt_question
        st.session_state.prebuilt_questions.remove(st.session_state.prebuilt_question)
        st.session_state.prebuilt_question = "" 
    else:
        user_input = st.session_state.get('content', "").strip()

    if not user_input:
        return

    st.session_state["messages"].append({"role": "user", "content": user_input})

    if st.session_state["use_feedback_graph"] and st.session_state["graph_fb"] is not None:
        graph = st.session_state["graph_fb"]
        try:
            response_obj = graph.invoke(
                {"summary": st.session_state["summary"], "qa_list": st.session_state["custom_qas"],
                 "questions": [user_input]})
            response = response_obj["response"]
        except Exception as e:
            response = f"Error generating feedback: {e}"

    elif st.session_state["use_qa_graph"] and st.session_state["graph_qa"] is not None:
        graph = st.session_state["graph_qa"]
        try:
            response_obj = graph.invoke({"questions": [user_input]},
                                        config=st.session_state["config"])
            if "end_responses" in response_obj and response_obj["end_responses"]:
                response = response_obj["end_responses"][-1]
            else:
                response = "No response found."
        except Exception as e:
            response = f"Error generating response: {e}"

    else:
        graph = st.session_state["graph_sc"]
        try:
            response_obj = graph.invoke({"messages": [user_input]})
            response = response_obj["messages"][-1].content
        except Exception as e:
            response = f"Error generating response: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": response})


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "graph_sc" not in st.session_state:
    st.session_state["graph_sc"] = simple_conversation_initialization()

if "use_qa_graph" not in st.session_state:
    st.session_state["use_qa_graph"] = False

if "graph_qa" not in st.session_state:
    st.session_state["graph_qa"] = None
    st.session_state["config"] = {"configurable": {"thread_id": str(uuid.uuid4())}}

if "custom_qas" not in st.session_state:
    st.session_state["custom_qas"] = []

if "graph_fb" not in st.session_state:
    st.session_state["graph_fb"] = None

if "use_feedback_graph" not in st.session_state:
    st.session_state["use_feedback_graph"] = False

if "prebuilt_questions" not in st.session_state:
    st.session_state.prebuilt_questions = [
        "Is the title concise, informative, and accurately reflects the paper's content?",
        "Have all authors listed appropriately and contributed significantly to the research?",
        "Is the methodology for detecting money laundering clearly and concisely described?",
        "Are the findings of the research adequately discussed and analyzed?",
        "Does the paper provide a clear and insightful outlook for future research in this area?"
    ]

st.title("Hey there! I'm Scholarly. Ready to review your paper and give you feedback. Let’s get started!")

uploaded_file = st.file_uploader('Upload your paper in .pdf format', type="pdf")
if uploaded_file is not None:
    temp_file_path = "/tmp/uploaded_paper.pdf"
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        st.error(f"Error saving the uploaded file: {e}")
    else:
        st.info("Processing a document... Please wait while I upload it to the database.")
        try:
            if st.session_state["graph_qa"] is None:
                st.session_state["graph_qa"], st.session_state["summary"] = qa_initialization(temp_file_path)
            if not st.session_state["use_feedback_graph"]:
                st.session_state["use_qa_graph"] = True
                st.session_state["use_feedback_graph"] = False
            st.success("It's done! Now you can ask questions about the uploaded document.")
        except Exception as e:
            st.error(f"Error initializing the QA graph: {e}")
            st.session_state["use_qa_graph"] = False

col_left, col_right = st.columns([3, 1])

with col_left:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.write("Suggested Questions:")
    cols = st.columns(3)
    for i, question in enumerate(st.session_state.prebuilt_questions):
        with cols[i % 3]:
            if st.button(question, key=f"prebuilt_q{i}"):
                st.session_state.prebuilt_question = question
                chat_content()
                st.rerun()

    with st.container():
        st.chat_input(
            "Hi! What do you want to ask?",
            key='content',
            on_submit=chat_content
        )

        button_b_pos = "0rem"
        button_css = float_css_helper(width="2.2rem", bottom=button_b_pos, transition=0)
        float_parent(css=button_css)

with col_right:
    st.subheader("Final Feedback")
    if st.button("Generate Feedback"):
        st.session_state["graph_fb"] = final_feedback_conversation_initialization()
        st.session_state["use_feedback_graph"] = True
        st.session_state["use_qa_graph"] = False
        st.success("Feedback graph initialized. Now all questions go to the Final Feedback mode!")

    st.subheader("Add Your Own Q&A Pair")
    user_question_input = st.text_input("Question")
    user_answer_input = st.text_area("Answer", height=100)

    if st.button("Save Q&A"):
        if user_question_input.strip() and user_answer_input.strip():
            st.session_state["custom_qas"].append({
                "question": user_question_input,
                "answer": user_answer_input
            })
            print(st.session_state["custom_qas"])
            st.success("Your Q&A pair is saved!")
        else:
            st.warning("Please fill both Question and Answer before saving.")

    if st.session_state["custom_qas"]:
        st.write("### Saved Q&A Pairs:")
        for i, pair in enumerate(st.session_state["custom_qas"], start=1):
            st.markdown(
                f"**{i}.** **Q**: {pair['question']}  \n"
                f"**A**: {pair['answer']}"
            )
