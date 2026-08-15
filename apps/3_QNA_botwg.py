from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# -----------------------------
# LLM
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    streaming=True
)


# -----------------------------
# Google Search
# -----------------------------

search = GoogleSerperAPIWrapper()

search_tool = Tool(
    name="google_search",
    description="Search Google for information.",
    func=search.run
)

tools = [search_tool]


# -----------------------------
# Memory
# -----------------------------

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()

if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# Agent
# -----------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are an amazing AI agent and can search on Google as well."
)


# -----------------------------
# Web Interface
# -----------------------------

st.subheader("QuickAnswer - Answers at the speed of thought")


# Show chat history
for message in st.session_state.history:
    role = message["role"]
    content = message["content"]

    st.chat_message(role).markdown(content)


# -----------------------------
# User Input
# -----------------------------

query = st.chat_input("Ask Anything?")


if query:

    # Show user message
    st.chat_message("user").markdown(query)

    # Save user message
    st.session_state.history.append({
        "role": "user",
        "content": query
    })


    # -----------------------------
    # Agent Stream
    # -----------------------------

    response = agent.stream(
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
        },
        stream_mode="messages"
    )


    # -----------------------------
    # Display AI Response
    # -----------------------------

    ai_container = st.chat_message("assistant")

    with ai_container:

        space = st.empty()
        message = ""

        for chunk, metadata in response:

            content = chunk.content

            # Gemini can sometimes return a list of content blocks
            if isinstance(content, str):

                text = content

            elif isinstance(content, list):

                text = ""

                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text += item.get("text", "")

            else:
                text = str(content)

            if text:
                message += text
                space.markdown(message)


        # Save final AI response
        st.session_state.history.append({
            "role": "assistant",
            "content": message
        })