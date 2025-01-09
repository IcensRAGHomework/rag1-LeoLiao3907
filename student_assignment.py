import base64
import datetime
import requests

from model_configurations import get_model_configuration
from model_creator import create_model

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor, Tool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

gpt_chat_version = 'gpt-4o' # Azure OpenAI gpt-4o
gpt_config = get_model_configuration(gpt_chat_version)

openai_gpt_chat_version = 'openai-gpt-4o'
gemini_chat_version = 'gemini-1.5-flash'
ACTIVE_MODEL = gpt_chat_version

CALENDARIFIC_API_URL = "https://calendarific.com/api/v2/holidays"
CALENDARIFIC_API_KEY = "I3eD2908aTUhgcXGaprER5cVKimNIttt"

@tool
def get_holidays(country: str, year: int = None) -> str:
    """
    Retrieves holidays for a given country and year using the Calendarific API.
    Input should be in the format 'country_code,year' e.g. 'US,2024'.
    If year is omitted, it defaults to the current year.
    """

    # Spliting the country and year
    if "," in country:
        country, year_str = country.split(",")
        try:
            year = int(year_str)
        except ValueError:
            return "Error: Invalid year format. Please use 'country_code,year' (e.g., 'US,2024')."
    # Bounding check for the year
    if year is None:
        year = datetime.datetime.now().year
    if year > 2049:
        return "Error: Calendarific API only supports years up to 2049."

    # Query holidays by the given country and year through Calendarific API
    try:
        params = {
            "api_key": CALENDARIFIC_API_KEY,
            "country": country,
            "year": year,
        }
        response = requests.get(CALENDARIFIC_API_URL, params = params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json() # Parse JSON response
		
		# Parse the result according to the API documentation available at
		# https://calendarific.com/api-documentation
        if data["meta"]["code"] == 200: # Retrieve the result code, 200: API call success
            holidays = data["response"]["holidays"] # Retrieve the holidays
            if not holidays:
                return f"No holidays found for {country} in {year}."

            # Format each holiday as "- name (date)" on a separate line
            return "\n".join([f"- {h['name']} ({h['date']['iso']})" for h in holidays])
        else:
            return f"API Error: {data['meta']['code']} - {data['meta']['error_type']}"
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Calendarific API: {e}"
    except (KeyError, TypeError) as e:
        return f"Error parsing Calendarific API response: {e}. \
            Raw Response: {response.text if 'response' in locals() else 'No response received'}"

def generate_hw01(question):
    system_prompt_str = """
        You are a helpful assistant who provides clear and accurate information.
        When a query relates to holidays, you will deliver dates and names in structured JSON format.
        The JSON structure should look like this:
        {{
            "Result": [
                {{
                    "date": "2024-12-25",
                    "name": "Christmas"
                }}
            ]
        }}
        Ensure that the output is in valid JSON format without any additional text likes backticks.

        The language of the output result will be translated 
        to match the language of the input string and keep the tags of JSON
        in English.
    """

    messages = [
        SystemMessage(content = [{"type": "text", "text": system_prompt_str},]),
        HumanMessage(content = [{"type": "text", "text": question},]),
    ]

    model = create_model(ACTIVE_MODEL)
    response = model.invoke(messages)

    result = response.content.strip().removeprefix("```").removesuffix("```")
    return result
    
def generate_hw02(question):
    agent_prompt_str = """
        You are a helpful assistant to identify the country and year 
        from a given query about holidays and call the tool `get_holidays` 
        with the extracted information.

        For example, if given the question "What are the holidays in the US in 2024?",
        you should identify the country as US and the year as 2024, 
        and then call the tool with these details: `get_holidays(country="US", year=2024)`.

        When providing the final answer to the user, format it as a structured JSON
        which date field for holiday date and name field for holiday name:
        {{
            "Result": [
                {{
                    "date": "2024-12-25",
                    "name": "Christmas"
                }}
            ]
        }}
        Ensure that the output is in valid JSON format without any additional text likes backticks.

        The language of the output result will be translated 
        to match the language of the input string and keep the tags of JSON
        in English.
    """
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", agent_prompt_str),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tools = [
        Tool(
            name = get_holidays.name,
            func = get_holidays,
            description = get_holidays.description,
            parameters = [
                ("country", str),
                ("year", int),
            ]
        )
    ]

    model = create_model(ACTIVE_MODEL)
    agent = create_tool_calling_agent(model, tools, agent_prompt)
    agent_executor = AgentExecutor(agent = agent, tools = tools)
    response = agent_executor.invoke({"input": question})

    result = response["output"].strip().removeprefix("```").removesuffix("```")
    print(result)
    return result
    
def generate_hw03(question2, question3):
    agent_prompt_str = """
        You are a helpful assistant to identify the country and year 
        from a given query about holidays and call the tool `get_holidays` 
        with the extracted information.

        If the given question is about "what are the holidays in the US in 2024?",
        you should identify the country as US and the year as 2024, 
        then call the tool with these details: `get_holidays(country="US", year=2024)`.
        And providing the answer to the user, format it as a structured JSON
        which date field for holiday date and name field for holiday name:
        {{
            "Result": [
                {{
                    "date": "2024-12-25",
                    "name": "Christmas"
                }}
            ]
        }}
        Ensure that the output is in valid JSON format without any additional text likes backticks.

        If the given question is about "is the given holiday in the list?",
        you should check the list and determine if the holiday exists 
        based on BOTH the date AND the name.
        If a holiday with the SAME DATE but a DIFFERENT NAME is provided, 
        it is considered a NEW holiday and recommended for addition.
        Provide the answer to the user, formatted as structured JSON 
        with the following structure:
        {{
            "Result": {{
                "add": true/false,
                "reason": "Describe why you do or do not want to add a new holiday, 
                        specify whether the holiday already exists in the list, 
                        and shows all the contents of the current list for reference."
            }}
        }}
        Ensure that the output is in valid JSON format without any additional text likes backticks.

        The language of the output result will be translated 
        to match the language of the input string and keep the tags of JSON
        in English.
    """
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", agent_prompt_str),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tools = [
        Tool(
            name = get_holidays.name,
            func = get_holidays,
            description = get_holidays.description,
            parameters = [
                ("country", str),
                ("year", int),
            ]
        )
    ]

    model = create_model(ACTIVE_MODEL)
    agent = create_tool_calling_agent(model, tools, agent_prompt)
    agent_executor = AgentExecutor(agent = agent, tools = tools)

    # Create chat history
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key = "input",
        history_messages_key = "chat_history",
    )

    session_id = "query_holiday"
    agent_with_chat_history.invoke(
        {"input": question2},
        config = {"configurable": {"session_id": session_id}},
    )
    response = agent_with_chat_history.invoke(
        {"input": question3},
        config = {"configurable": {"session_id": session_id}},
    )

    result = response["output"].strip().removeprefix("```").removesuffix("```")
    print(result)
    return result
    
def generate_hw04(question):
    def encode_to_image_url(image_path, image_type = "jpeg"):
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                return f"data:image/{image_type};base64,{encoded_string}"  # Correct MIME type
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
    base64_image = encode_to_image_url("baseball.png")

    system_prompt_str = """
        You are a helpful assistant to provide clear and accurate information.
        When a query is about a match score, you must find the information from the image
        and provide the results strictly as structured JSON like below:
        {{
            "Result": {{
                "score": 1234
            }}
        }}
        Ensure that the output is in valid JSON format without any additional text likes backticks.

        The language of the output result will be translated 
        to match the language of the input string and keep the tags of JSON
        in English.
    """

    messages = [
        SystemMessage(content = [{"type": "text", "text": system_prompt_str},]),
        HumanMessage(content = [
            {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                },
            ],
        ),
    ]

    model = create_model(ACTIVE_MODEL)
    response = model.invoke(messages)

    result = response.content.strip().removeprefix("```").removesuffix("```")
    print(result)
    return result
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
