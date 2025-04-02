# -*- coding = utf-8 -*-
# @time:2024/7/24 16:42
# Author:david yuan
# @File:conversable_agent.py
# @Software:VeSync


import asyncio
import copy
import functools
import inspect
import json
import re
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from vagents.instance.base import LLMAgent,Agent
from vagents.vagentic.llms.client import OpenAIClient


class ConversableAgent(LLMAgent):

    DEFAULT_CONFIG = False  # False or dict, the default config for llm inference
    MAX_CONSECUTIVE_AUTO_REPLY = 100  # maximum number of consecutive auto replies (subject to future change)

    DEFAULT_SUMMARY_PROMPT = "Summarize the takeaway from the conversation. Do not add any introductory phrases."
    DEFAULT_SUMMARY_METHOD = "last_msg"
    llm_config: Union[Dict, Literal[False]]

    def __init__(
        self,
        name: str,
        system_message: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
    ):

        self._name = name
        if chat_messages is None:
            self._oai_messages = defaultdict(list)
        else:
            self._oai_messages = chat_messages

        self._oai_system_message = [{"content": system_message, "role": "system"}]

        if isinstance(llm_config, dict):
            try:
                llm_config = copy.deepcopy(llm_config)
            except TypeError as e:
                raise TypeError(
                    "Please implement __deepcopy__ method for each value class in llm_config to support deepcopy."
                    " Refer to the docs for more details: https://microsoft.github.io/autogen/docs/topics/llm_configuration#adding-http-client-in-llm_config-for-proxy"
                ) from e

        self.client = OpenAIClient(llm_config)
        '''
        注册函数
        '''
        self._reply_func_list = []
        self.register_reply([Agent, None], ConversableAgent.generate_oai_reply)
        self.register_reply([Agent, None], ConversableAgent.a_generate_oai_reply, ignore_async_in_sync_chat=True)

    def replace_reply_func(self, old_reply_func: Callable, new_reply_func: Callable):
        for f in self._reply_func_list:
            if f["reply_func"] == old_reply_func:
                f["reply_func"] = new_reply_func


    @property
    def system_message(self) -> str:
        """Return the system message."""
        return self._oai_system_message[0]["content"]

    def update_system_message(self, system_message: str) -> None:
        """Update the system message.

        Args:
            system_message (str): system message for the ChatCompletion inference.
        """
        self._oai_system_message[0]["content"] = system_message


    @property
    def chat_messages(self) -> Dict[Agent, List[Dict]]:
        """A dictionary of conversations from agent to list of messages."""
        return self._oai_messages

    def chat_messages_for_agent(self, agent: Agent) -> List[Dict]:
        """A list of messages as a conversation to summarize."""
        return self._oai_messages[agent]

    def last_message_for_agent(self, agent: Optional[Agent] = None) -> Optional[Dict]:
        """The last message exchanged with the agent.

        Args:
            agent (Agent): The agent in the conversation.
                If None and more than one agent's conversations are found, an error will be raised.
                If None and only one conversation is found, the last message of the only conversation will be returned.

        Returns:
            The last message exchanged with the agent.
        """
        if agent is None:
            n_conversations = len(self._oai_messages)
            if n_conversations == 0:
                return None
            if n_conversations == 1:
                for conversation in self._oai_messages.values():
                    return conversation[-1]
            raise ValueError("More than one conversation is found. Please specify the sender to get the last message.")
        if agent not in self._oai_messages.keys():
            raise KeyError(
                f"The agent '{agent.name}' is not present in any conversation. No history available for this agent."
            )
        return self._oai_messages[agent][-1]


    @staticmethod
    def _message_to_dict(message: Union[Dict, str]) -> Dict:
        """Convert a message to a dictionary.

        The message can be a string or a dictionary. The string will be put in the "content" field of the new dictionary.
        """
        if isinstance(message, str):
            return {"content": message}
        elif isinstance(message, dict):
            return message
        else:
            return dict(message)

    @staticmethod
    def _normalize_name(name):
        """
        LLMs sometimes ask functions while ignoring their own format requirements, this function should be used to replace invalid characters with "_".

        Prefer _assert_valid_name for validating user configuration or input
        """
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]

    @staticmethod
    def _assert_valid_name(name):
        """
        Ensure that configured names are valid, raises ValueError if not.

        For munging LLM responses use _normalize_name to ensure LLM specified names don't break the API.
        """
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(f"Invalid name: {name}. Only letters, numbers, '_' and '-' are allowed.")
        if len(name) > 64:
            raise ValueError(f"Invalid name: {name}. Name must be less than 64 characters.")
        return name

    def _append_oai_message(self, message: Union[Dict, str], role, conversation_id: Agent) -> bool:
        """Append a message to the ChatCompletion conversation.

        If the message received is a string, it will be put in the "content" field of the new dictionary.
        If the message received is a dictionary but does not have any of the three fields "content", "function_call", or "tool_calls",
            this message is not a valid ChatCompletion message.
        If only "function_call" or "tool_calls" is provided, "content" will be set to None if not provided, and the role of the message will be forced "assistant".

        Args:
            message (dict or str): message to be appended to the ChatCompletion conversation.
            role (str): role of the message, can be "assistant" or "function".
            conversation_id (Agent): id of the conversation, should be the recipient or sender.

        Returns:
            bool: whether the message is appended to the ChatCompletion conversation.
        """
        message = self._message_to_dict(message)
        # create oai message to be appended to the oai conversation that can be passed to oai directly.
        oai_message = {
            k: message[k]
            for k in ("content", "function_call", "tool_calls", "tool_responses", "tool_call_id", "name", "context")
            if k in message and message[k] is not None
        }
        if "content" not in oai_message:
            if "function_call" in oai_message or "tool_calls" in oai_message:
                oai_message["content"] = None  # if only function_call is provided, content will be set to None.
            else:
                return False

        if message.get("role") in ["function", "tool"]:
            oai_message["role"] = message.get("role")
        elif "override_role" in message:
            # If we have a direction to override the role then set the
            # role accordingly. Used to customise the role for the
            # select speaker prompt.
            oai_message["role"] = message.get("override_role")
        else:
            oai_message["role"] = role

        if oai_message.get("function_call", False) or oai_message.get("tool_calls", False):
            oai_message["role"] = "assistant"  # only messages with role 'assistant' can have a function call.
        self._oai_messages[conversation_id].append(oai_message)
        return True


    async def a_send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):

        # When the agent composes and sends the message, the role of the message is "assistant"
        # unless it's "function".
        valid = self._append_oai_message(message, "assistant", recipient)
        if valid:
            await recipient.a_receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )


    def _print_received_message(self, message: Union[Dict, str], sender: Agent):
        iostream = IOStream.get_default()
        # print the message received
        iostream.print(colored(sender.name, "yellow"), "(to", f"{self.name}):\n", flush=True)
        message = self._message_to_dict(message)

        if message.get("tool_responses"):  # Handle tool multi-call responses
            for tool_response in message["tool_responses"]:
                self._print_received_message(tool_response, sender)
            if message.get("role") == "tool":
                return  # If role is tool, then content is just a concatenation of all tool_responses

        if message.get("role") in ["function", "tool"]:
            if message["role"] == "function":
                id_key = "name"
            else:
                id_key = "tool_call_id"
            id = message.get(id_key, "No id found")
            func_print = f"***** Response from calling {message['role']} ({id}) *****"
            iostream.print(colored(func_print, "green"), flush=True)
            iostream.print(message["content"], flush=True)
            iostream.print(colored("*" * len(func_print), "green"), flush=True)
        else:
            content = message.get("content")
            if content is not None:
                if "context" in message:
                    content = message["context"]
                iostream.print(content, flush=True)
            if "function_call" in message and message["function_call"]:
                function_call = dict(message["function_call"])
                func_print = (
                    f"***** Suggested function call: {function_call.get('name', '(No function name found)')} *****"
                )
                iostream.print(colored(func_print, "green"), flush=True)
                iostream.print(
                    "Arguments: \n",
                    function_call.get("arguments", "(No arguments found)"),
                    flush=True,
                    sep="",
                )
                iostream.print(colored("*" * len(func_print), "green"), flush=True)
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    id = tool_call.get("id", "No tool call id found")
                    function_call = dict(tool_call.get("function", {}))
                    func_print = f"***** Suggested tool call ({id}): {function_call.get('name', '(No function name found)')} *****"
                    iostream.print(colored(func_print, "green"), flush=True)
                    iostream.print(
                        "Arguments: \n",
                        function_call.get("arguments", "(No arguments found)"),
                        flush=True,
                        sep="",
                    )
                    iostream.print(colored("*" * len(func_print), "green"), flush=True)

        iostream.print("\n", "-" * 80, flush=True, sep="")



    def _process_received_message(self, message: Union[Dict, str], sender: Agent, silent: bool):
        # When the agent receives a message, the role of the message is "user". (If 'role' exists and is 'function', it will remain unchanged.)
        valid = self._append_oai_message(message, "user", sender)

        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )
        if not silent:
            self._print_received_message(message, sender)


    '''
    这个函数无需和大模型交互，入口函数
    '''
    def send(
            self,
            message: Union[Dict, str],
            recipient: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
    ):
        valid = self._append_oai_message(message, "assistant", recipient)
        if valid:
            recipient.receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        valid = self._append_oai_message(message, "user", sender)
        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

        if not silent:
            self._print_received_message(message, sender)

        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return

        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(reply, sender, silent=silent)


    '''
    人-agent交互
    '''

    def chat(self,
             message: Union[Dict, str],
             sender: Agent,
             silent: Optional[bool] = False):

        try:
            valid = self._append_oai_message(message, "user", sender)
            if not valid:
                raise ValueError(
                    "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided.")

            print(f'messages ====>: {self.chat_messages[sender]}')
            reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
            if reply is not None:
                valid = self._append_oai_message(reply, "assistant", sender)
                if valid:
                    return reply
        except Exception as e:
            print(f"An error occurred: {e}")

    async def a_receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        self._process_received_message(message, sender, silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = await self.a_generate_reply(sender=sender)
        if reply is not None:
            await self.a_send(reply, sender, silent=silent)



    def generate_oai_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai."""
        client = self.client
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]
        extracted_response = self._generate_oai_reply_from_client(
            client, self._oai_system_message + messages
        )
        return (False, None) if extracted_response is None else (True, extracted_response)


    '''
    实际的调用llm
    '''
    def _generate_oai_reply_from_client(self, llm_client, messages) -> Union[str, Dict, None]:
        # unroll tool_responses
        all_messages = []
        for message in messages:
            tool_responses = message.get("tool_responses", [])
            if tool_responses:
                all_messages += tool_responses
                if message.get("role") != "tool":
                    all_messages.append({key: message[key] for key in message if key != "tool_responses"})
            else:
                all_messages.append(message)

        response = llm_client.create(
            messages=all_messages
        )
        extracted_response = llm_client.message_retrieval(response)[0]
        if extracted_response is None:
            warnings.warn(f"Extracted_response from {response} is None.", UserWarning)
            return None

        if not isinstance(extracted_response, str) and hasattr(extracted_response, "model_dump"):
            extracted_response = model_dump(extracted_response)
        if isinstance(extracted_response, dict):
            if extracted_response.get("function_call"):
                extracted_response["function_call"]["name"] = self._normalize_name(
                    extracted_response["function_call"]["name"]
                )
            for tool_call in extracted_response.get("tool_calls") or []:
                tool_call["function"]["name"] = self._normalize_name(tool_call["function"]["name"])
                # Remove id and type if they are not present.
                # This is to make the tool call object compatible with Mistral API.
                if tool_call.get("id") is None:
                    tool_call.pop("id")
                if tool_call.get("type") is None:
                    tool_call.pop("type")
        return extracted_response



    async def a_generate_oai_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai asynchronously."""
        iostream = IOStream.get_default()

        def _generate_oai_reply(
            self, iostream: IOStream, *args: Any, **kwargs: Any
        ) -> Tuple[bool, Union[str, Dict, None]]:
            with IOStream.set_default(iostream):
                return self.generate_oai_reply(*args, **kwargs)

        return await asyncio.get_event_loop().run_in_executor(
            None,
            functools.partial(
                _generate_oai_reply, self=self, iostream=iostream, messages=messages, sender=sender, config=config
            ),
        )

    def generate_reply(
            self,
            messages: Optional[List[Dict[str, Any]]] = None,
            sender: Optional["Agent"] = None,
            **kwargs: Any,
    ) -> Union[str, Dict, None]:

        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        success, reply = self.generate_oai_reply(messages=messages, sender=sender)
        if success:
            return reply
        else:
            print("Failed to generate reply.")
            return None

    def _match_trigger(self, trigger: Union[None, str, type, Agent, Callable, List], sender: Optional[Agent]) -> bool:
        if trigger is None:
            return sender is None
        elif isinstance(trigger, str):
            if sender is None:
                raise SenderRequired()
            return trigger == sender.name
        elif isinstance(trigger, type):
            return isinstance(sender, trigger)
        elif isinstance(trigger, Agent):
            # return True if the sender is the same type (class) as the trigger
            return trigger == sender
        elif isinstance(trigger, Callable):
            rst = trigger(sender)
            assert isinstance(rst, bool), f"trigger {trigger} must return a boolean value."
            return rst
        elif isinstance(trigger, list):
            return any(self._match_trigger(t, sender) for t in trigger)
        else:
            raise ValueError(f"Unsupported trigger type: {type(trigger)}")


    async def a_generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional["Agent"] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], None]:

        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]


        # Call the hookable method that gives registered hooks a chance to process the last message.
        # Message modifications do not affect the incoming messages or self._oai_messages.
        messages = self.process_last_received_message(messages)

        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if "exclude" in kwargs and reply_func in kwargs["exclude"]:
                continue

            if self._match_trigger(reply_func_tuple["trigger"], sender):
                if inspect.iscoroutinefunction(reply_func):
                    final, reply = await reply_func(
                        self, messages=messages, sender=sender, config=reply_func_tuple["config"]
                    )
                else:
                    final, reply = reply_func(self, messages=messages, sender=sender, config=reply_func_tuple["config"])
                if final:
                    return reply
        return self._default_auto_reply



    def process_last_received_message(self, messages: List[Dict]) -> List[Dict]:
        """
        Calls any registered capability hooks to use and potentially modify the text of the last message,
        as long as the last message is not a function call or exit command.
        """

        # If any required condition is not met, return the original message list.
        hook_list = self.hook_lists["process_last_received_message"]
        if len(hook_list) == 0:
            return messages  # No hooks registered.
        if messages is None:
            return None  # No message to process.
        if len(messages) == 0:
            return messages  # No message to process.
        last_message = messages[-1]
        if "function_call" in last_message:
            return messages  # Last message is a function call.
        if "context" in last_message:
            return messages  # Last message contains a context key.
        if "content" not in last_message:
            return messages  # Last message has no content.

        user_content = last_message["content"]
        if not isinstance(user_content, str) and not isinstance(user_content, list):
            # if the user_content is a string, it is for regular LLM
            # if the user_content is a list, it should follow the multimodal LMM format.
            return messages
        if user_content == "exit":
            return messages  # Last message is an exit command.

        # Call each hook (in order of registration) to process the user's message.
        processed_user_content = user_content
        for hook in hook_list:
            processed_user_content = hook(processed_user_content)

        if processed_user_content == user_content:
            return messages  # No hooks actually modified the user's message.

        # Replace the last user message with the expanded one.
        messages = messages.copy()
        messages[-1]["content"] = processed_user_content
        return messages






