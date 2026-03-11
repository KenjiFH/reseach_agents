import streamlit as st
from document_processor import process_document_and_create_vdb
from agents import create_multi_agent_workflow

st.set_page_config(page_title="Multi-Agent RAG POC", page_icon="🤖", layout="centered")
st.title("🤖 Multi-Agent RAG Assistant")

if "workflow" not in st.session_state:
    st.session_state.workflow = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("1. Upload Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    # Condition 1: A file is uploaded, but agents aren't built yet
    if uploaded_file and st.session_state.workflow is None:
        with st.spinner("Processing document and building Vector DB..."):
            retriever = process_document_and_create_vdb(uploaded_file)
            st.session_state.workflow = create_multi_agent_workflow(retriever)
        st.success("Document processed! Agents are ready.")
    # Condition 2 (NEW): The file was removed (X clicked), but agents are still in memory
    elif uploaded_file is None and st.session_state.workflow is not None:
        st.session_state.workflow = None
        st.session_state.messages = []
        st.rerun() # Immediately refresh the app to clear the screen
    # Condition 3: The manual reset button
    if st.button("Clear Chat & Reset"):
        st.session_state.workflow = None
        st.session_state.messages = []
        st.rerun()

st.header("2. Ask the Agents")

# Display existing chat messages WITH their metrics
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If the message is from the assistant and has scores, display them
        if msg["role"] == "assistant" and "metrics" in msg:
            col1, col2 = st.columns(2)
            col1.metric("🔍 Faithfulness", f"{msg['metrics']['faithfulness']}%")
            col2.metric("🎯 Relevance", f"{msg['metrics']['relevance']}%")

if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.workflow is None:
        st.error("Please upload a document in the sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agents are researching, synthesizing, and evaluating..."):
                
                initial_state = {
                    "question": prompt, 
                    "context": "", 
                    "answer": "", 
                    "faithfulness_score": 0, 
                    "relevance_score": 0
                }
                
                final_state = st.session_state.workflow.invoke(initial_state)
                
                response = final_state["answer"]
                f_score = final_state["faithfulness_score"]
                r_score = final_state["relevance_score"]
                
                # Display the answer
                st.markdown(response)
                
                # Display the metrics beautifully
                col1, col2 = st.columns(2)
                col1.metric("🔍 Faithfulness (Hallucination Check)", f"{f_score}%")
                col2.metric("🎯 Relevance (Accuracy Check)", f"{r_score}%")
                
        # Save both the content and the metrics to the session state history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "metrics": {"faithfulness": f_score, "relevance": r_score}
        })