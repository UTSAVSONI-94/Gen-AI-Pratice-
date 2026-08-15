import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Database
# ---------------------------------------------------------
db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

db.run("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK (
            status IN ('pending', 'in_progress', 'completed')
        ) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")


# ---------------------------------------------------------
# Gemini model
# ---------------------------------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
)


# ---------------------------------------------------------
# SQL tools
# ---------------------------------------------------------
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model
)

tools = toolkit.get_tools()


# ---------------------------------------------------------
# System prompt
# ---------------------------------------------------------
system_prompt = """
You are a task management assistant that interacts with a SQL database
containing a 'tasks' table.

DATABASE SCHEMA:
tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK (
        status IN ('pending', 'in_progress', 'completed')
    ),
    created_at TIMESTAMP
)

TASK RULES:

1. For SELECT queries:
   - Return a maximum of 10 results.
   - Always use ORDER BY created_at DESC unless the user explicitly
     requests a different ordering.

2. After CREATE, UPDATE, or DELETE:
   - Always perform a SELECT query to verify the operation.
   - Tell the user whether the operation was successful.

3. CREATE:
   - Use INSERT INTO tasks(title, description, status).

4. READ:
   - Use SELECT queries.
   - Limit results to 10.

5. UPDATE:
   - Use UPDATE tasks SET ...
   - The task can be identified by id or title.

6. DELETE:
   - Use DELETE FROM tasks WHERE ...
   - The task can be identified by id or title.

7. Status must be one of:
   - pending
   - in_progress
   - completed

8. Never invent task IDs, titles, descriptions, statuses, or
   database results.

9. If the user asks to list/show/view tasks, present the result
   as a clean Markdown table.

10. Before executing destructive operations such as DELETE, make
    sure you have enough information to identify the intended task.
"""


# ---------------------------------------------------------
# Create agent
# ---------------------------------------------------------
@st.cache_resource
def get_agent():
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )

    return agent


agent = get_agent()


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(
    page_title="TaskBot",
    page_icon="📜",
    layout="centered",
)

st.title("📜 TaskBot")
st.caption("Manage your tasks using natural language")


# ---------------------------------------------------------
# Chat history for displaying messages
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------
prompt = st.chat_input(
    "Ask me to manage your tasks..."
)


if prompt:

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Agent response
    with st.chat_message("assistant"):

        with st.spinner("Processing..."):

            try:

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": "taskbot-user-1"
                        }
                    },
                )

                result = response["messages"][-1].content

                st.markdown(result)

                # Store assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result,
                    }
                )

            except Exception as e:

                error_message = f"❌ Error: {str(e)}"

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )