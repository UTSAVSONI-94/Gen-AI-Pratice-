from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

search = GoogleSerperAPIWrapper()

memory = MemorySaver()

agent = create_agent(
    model=model,
    tools=[search.run],
    checkpointer=memory,
    system_prompt="You are an agent and can search Google for any question."
)

while True:
    query = input("User: ")

    if query.lower() == "quit":
        print("Good Bye 👋")
        break

    response = agent.invoke(
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

    print("AI:", response["messages"][-1].content)