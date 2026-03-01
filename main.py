import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
# from pydantic_core.core_schema import model_field
from todoist_api_python.api import TodoistAPI

load_dotenv()

TODOIST_API_KEY = os.getenv("TODOIST_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

todoist_api = TodoistAPI(TODOIST_API_KEY)

@tool
def add_task(task: str, description: str = None):
    """Add a new task to the user's task list. 
    Use this when the user wants to add a new task"""

    todoist_api.add_task(content=task, description=description)


tools = [add_task]

llm = ChatGoogleGenerativeAI(
  model = 'gemini-2.5-flash',
  google_api_key = GEMINI_API_KEY,
  temperature=0.3
)

system_prompt = "You are a helpful AI assistant, You will help user manage their tasks in Todoist platform."

agent = create_agent(llm, tools=tools, system_prompt=system_prompt)

while True:
    user_input = input('You: ')
    response = agent.invoke({"messages": [HumanMessage(user_input)]})

    # Get the final AI response
    final_message = response["messages"][-1].content

    print(final_message)