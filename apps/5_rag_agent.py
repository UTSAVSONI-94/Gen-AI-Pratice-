from dotenv import load_dotenv
load_dotenv()

import os
import shutil
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="PDF RAG Agent",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# Session State
# ============================================================

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Process Documents
# ============================================================

def process_document(path):

    # --------------------------------------------------------
    # Load PDFs
    # --------------------------------------------------------

    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    if not docs:
        st.error("No PDF documents were found.")
        return

    # --------------------------------------------------------
    # Split documents into chunks
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(docs)

    # --------------------------------------------------------
    # Gemini Embeddings
    # --------------------------------------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # --------------------------------------------------------
    # Create Vector Store
    # --------------------------------------------------------

    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    st.session_state.vector_store = vector_db

    # --------------------------------------------------------
    # Gemini LLM
    # --------------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )

    # --------------------------------------------------------
    # Retrieval Tool
    # --------------------------------------------------------

    @tool
    def retrieve_context(query: str):
        """
        Retrieve relevant information from the uploaded
        PDF documents.
        """

        docs = vector_db.similarity_search(
            query=query,
            k=3
        )

        if not docs:
            return "No relevant information was found in the documents."

        # Combine all retrieved documents
        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        return context

    # --------------------------------------------------------
    # System Prompt
    # --------------------------------------------------------

    system_prompt = """
You are a helpful PDF question-answering assistant.

The knowledge base consists of the PDF documents uploaded by
the user.

IMPORTANT RULES:

1. ALWAYS use the retrieve_context tool when answering questions
   that require information from the uploaded documents.

2. Answer questions using the retrieved document context.

3. Do not make up information that is not present in the
   retrieved context.

4. If the answer cannot be found in the uploaded documents,
   clearly tell the user that the information is not available
   in the uploaded documents.

5. Give clear and concise answers.

6. When appropriate, explain the answer using information from
   the retrieved documents.
"""

    # --------------------------------------------------------
    # Agent Memory
    # --------------------------------------------------------

    memory = InMemorySaver()

    # --------------------------------------------------------
    # Create Agent
    # --------------------------------------------------------

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    # Save agent in session
    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("📚 PDF RAG Agent")

    st.markdown(
        """
        Upload one or more PDF files and ask questions
        about their content.
        """
    )

    st.divider()

    if st.session_state.document_uploaded:

        st.success("Documents loaded successfully.")

        if st.button("🗑️ Clear Documents"):

            st.session_state.document_uploaded = False
            st.session_state.agent = None
            st.session_state.vector_store = None
            st.session_state.messages = []

            # Remove uploaded files
            if os.path.exists("./doc_files"):
                shutil.rmtree("./doc_files")

            st.rerun()


# ============================================================
# Upload UI
# ============================================================

if not st.session_state.document_uploaded:

    st.title("📚 PDF RAG Agent")

    st.subheader("Upload your PDF documents")

    uploaded_files = st.file_uploader(
        label="Select PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        with st.spinner("Processing your documents..."):

            # ------------------------------------------------
            # Create directory
            # ------------------------------------------------

            path = "./doc_files"

            os.makedirs(path, exist_ok=True)

            # ------------------------------------------------
            # Save uploaded files
            # ------------------------------------------------

            for file in uploaded_files:

                file_path = os.path.join(
                    path,
                    file.name
                )

                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

            # ------------------------------------------------
            # Process documents
            # ------------------------------------------------

            process_document(path)

        if st.session_state.document_uploaded:

            st.success(
                f"{len(uploaded_files)} PDF file(s) processed successfully!"
            )

            st.rerun()


# ============================================================
# Chat UI
# ============================================================

if (
    st.session_state.document_uploaded
    and st.session_state.agent
):

    st.title("💬 Chat with your Documents")

    st.caption(
        "Ask questions about the uploaded PDF documents."
    )

    # --------------------------------------------------------
    # Display Previous Messages
    # --------------------------------------------------------

    for message in st.session_state.messages:

        role = message["role"]
        content = message["content"]

        st.chat_message(role).markdown(content)

    # --------------------------------------------------------
    # Chat Input
    # --------------------------------------------------------

    query = st.chat_input(
        "Ask anything about your documents..."
    )

    if query:

        # ----------------------------------------------------
        # Display User Message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        st.chat_message("user").markdown(query)

        # ----------------------------------------------------
        # Invoke Agent
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = st.session_state.agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": query
                            }
                        ]
                    },
                    {
                        "configurable": {
                            "thread_id": "1"
                        }
                    }
                )

                # ------------------------------------------------
                # Extract Final Answer
                # ------------------------------------------------

                answer = response["messages"][-1].content

                # ------------------------------------------------
                # Display Answer
                # ------------------------------------------------

                st.markdown(answer)

        # ----------------------------------------------------
        # Save Assistant Message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
