from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

st.title("🤖 AskBuddy – AI QnA Bot")
st.markdown("My QnA Bot with LangChain and Google Gemini!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    st.chat_message(role).markdown(content)

# Chat input
query = st.chat_input("Ask anything?")

if query:
    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.chat_message("user").markdown(query)

    # Get Gemini response
    res = llm.invoke(query)

    # IMPORTANT: use .text instead of .content
    answer = res.text

    # Display only the actual answer
    st.chat_message("ai").markdown(answer)

    # Save only the actual answer
    st.session_state.messages.append({
        "role": "ai",
        "content": answer
    })