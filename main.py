import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# from pydantic_core.core_schema import model_field

load_dotenv()

TODOIST_API_KEY = os.getenv("TODOIST_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
  model = 'gemini-2.5-flash',
  google_api_key = GEMINI_API_KEY,
  temperature=0.3
)

system_prompt = "You are a helpful AI assistant, You will help user manage their tasks in Todoist platform."
user_input = "Best way to learn python practically?"

prompt = ChatPromptTemplate([
  ("system", system_prompt), 
  ("user", user_input)
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"input": user_input})
print(response)