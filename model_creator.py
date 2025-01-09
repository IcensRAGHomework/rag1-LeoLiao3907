from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def create_model(config_name):
    config = get_model_configuration(config_name)

    if config_name == 'gpt-4o': # Azure OpenAI gpt-4o
        llm = AzureChatOpenAI(
            model = config['model_name'],
            deployment_name = config['deployment_name'],
            openai_api_key = config['api_key'],
            openai_api_version = config['api_version'],
            azure_endpoint = config['api_base'],
            temperature = config['temperature']
        )
    elif config_name == 'openai-gpt-4o':
        llm = ChatOpenAI(
            model = config['model_name'],
            openai_api_key = config['api_key'],
            temperature = config['temperature']
        )
    elif config_name == 'gemini-1.5-flash':
        llm = ChatGoogleGenerativeAI(
            model = config['model_name'],
            google_api_key = config['api_key'],
            temperature = config['temperature']
        )

    return llm
