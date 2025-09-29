import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("PLAYGROUND_AOAI_ENDPOINT")
api_token = os.getenv("PLAYGROUND_LLM_API_TOKEN")
deployment_name = os.getenv("AOAI_DEPLOY_GPT4O_MINI")

model = AzureChatOpenAI(
    openai_api_key='no-need-api-key-but-never-remove-this-line',
    azure_endpoint=endpoint,
    azure_deployment=deployment_name,
    api_version="2024-10-21",
    default_headers={
        "Authorization": f"Basic {api_token}"
    }
)

messages=[
    ("system", "You are a helpful assistant. Response in Korean."),
    ("human", "Do you know PSY?")
]

ai_msg = model.invoke(messages)
print(ai_msg.content)