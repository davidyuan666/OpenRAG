from openai import OpenAI
from openai.types.completion import Completion


class OpenAIClient:
    def __init__(self, llm_config):
        self.MODEL=llm_config['model']
        self.OPENAI_API_KEY = llm_config['api_key']
        self._oai_client =OpenAI(api_key=self.OPENAI_API_KEY)
        self.TOOL_ENABLED = False
        self.all_history = []

    '''
    get the message 
    '''
    def message_retrieval(self, response):
        """Retrieve the messages from the response."""
        messages = []
        for choice in response.choices:
            if isinstance(response, Completion):
                messages.append(choice.text)
            elif self.TOOL_ENABLED:
                if choice.message.function_call is not None or choice.message.tool_calls is not None:
                    messages.append(choice.message)
                else:
                    messages.append(choice.message.content)
            else:
                if choice.message.function_call is not None:
                    messages.append(choice.message)
                else:
                    messages.append(choice.message.content)
        return messages



    '''
    openai_response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [{'role': 'user', 'content': prompt_1}]
    )
    
    openai_response.choices[0].message.content
    '''
    def create(self,messages):
        """Create a completion for a given config using openai's client."""
        response = self._oai_client.chat.completions.create(model=self.MODEL,messages=messages,temperature=0)
        response_result =  response.choices[0].message
        return response_result


    def add_system_message(self,message):
        system_msg = [{'role': 'system', 'content': message}]
        response = self._oai_client.chat.completions.create(model=self.MODEL, messages=system_msg,temperature=0)
        response_result = response.choices[0].message.content
        return response_result


    def add_user_message(self,message):
        """Create a completion for a given config using openai's client."""
        user_msg = [{'role': 'user', 'content': message}]
        response = self._oai_client.chat.completions.create(model=self.MODEL, messages=user_msg,temperature=0)
        response_result = response.choices[0].message.content
        return response_result

    def start_chat(self,message):
        user_msg = [{'role': 'user', 'content': message}]
        response = self._oai_client.chat.completions.create(model=self.MODEL, messages=user_msg, temperature=0)
        reply_msg = response.choices[0].message.content
        assistant_msg =  [{'role': 'assistant', 'content': reply_msg}]
        self.all_history.append(user_msg)
        self.all_history.append(assistant_msg)
        return reply_msg


    def chat(self,message):
        user_msg = [{'role': 'user', 'content': message}]
        response = self._oai_client.chat.completions.create(model=self.MODEL, messages=user_msg, temperature=0)
        reply_msg = response.choices[0].message.content
        return reply_msg



