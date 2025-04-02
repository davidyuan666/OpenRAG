# -*- coding = utf-8 -*-
# @time:2024/7/29 10:15
# Author:david yuan
# @File:base.py
# @Software:VeSync


class LLM(BaseModel):
    model: str
    name: Optional[str] = None
    metrics: Dict[str, Any] = {}
    response_format: Optional[Any] = None
    tools: Optional[List[Union[Tool, Dict]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    run_tools: bool = True
    show_tool_calls: Optional[bool] = None
    functions: Optional[Dict[str, Function]] = None
    function_call_limit: int = 10
    function_call_stack: Optional[List[FunctionCall]] = None
    system_prompt: Optional[str] = None
    instructions: Optional[List[str]] = None
    run_id: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def api_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError

    def invoke(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def ainvoke(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def invoke_stream(self, *args, **kwargs) -> Iterator[Any]:
        raise NotImplementedError

    async def ainvoke_stream(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def response(self, messages: List[Message]) -> str:
        raise NotImplementedError

    async def aresponse(self, messages: List[Message]) -> str:
        raise NotImplementedError

    def response_stream(self, messages: List[Message]) -> Iterator[str]:
        raise NotImplementedError

    async def aresponse_stream(self, messages: List[Message]) -> Any:
        raise NotImplementedError

    def generate(self, messages: List[Message]) -> Dict:
        raise NotImplementedError

    def generate_stream(self, messages: List[Message]) -> Iterator[Dict]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(include={"name", "model", "metrics"})
        if self.functions:
            _dict["functions"] = {k: v.to_dict() for k, v in self.functions.items()}
            _dict["function_call_limit"] = self.function_call_limit
        return _dict

    def get_tools_for_api(self) -> Optional[List[Dict[str, Any]]]:
        if self.tools is None:
            return None

        tools_for_api = []
        for tool in self.tools:
            if isinstance(tool, Tool):
                tools_for_api.append(tool.to_dict())
            elif isinstance(tool, Dict):
                tools_for_api.append(tool)
        return tools_for_api

    def add_tool(self, tool: Union[Tool, Toolkit, Callable, Dict, Function]) -> None:
        if self.tools is None:
            self.tools = []

        # If the tool is a Tool or Dict, add it directly to the LLM
        if isinstance(tool, Tool) or isinstance(tool, Dict):
            self.tools.append(tool)
            logger.debug(f"Added tool {tool} to LLM.")
        # If the tool is a Callable or ToolRegistry, add its functions to the LLM
        elif callable(tool) or isinstance(tool, Toolkit) or isinstance(tool, Function):
            if self.functions is None:
                self.functions = {}

            if isinstance(tool, Toolkit):
                self.functions.update(tool.functions)
                for func in tool.functions.values():
                    self.tools.append({"type": "function", "function": func.to_dict()})
                logger.debug(f"Functions from {tool.name} added to LLM.")
            elif isinstance(tool, Function):
                self.functions[tool.name] = tool
                self.tools.append({"type": "function", "function": tool.to_dict()})
                logger.debug(f"Function {tool.name} added to LLM.")
            elif callable(tool):
                func = Function.from_callable(tool)
                self.functions[func.name] = func
                self.tools.append({"type": "function", "function": func.to_dict()})
                logger.debug(f"Function {func.name} added to LLM.")

    def deactivate_function_calls(self) -> None:
        self.tool_choice = "none"

    def run_function_calls(self, function_calls: List[FunctionCall], role: str = "tool") -> List[Message]:
        results: List[Message] = []
        for function_call in function_calls:
            if self.function_call_stack is None:
                self.function_call_stack = []

            # -*- Run function call
            t = Timer()
            t.start()
            function_call.execute()
            t.stop()
            m = Message(
                role=role,
                content=function_call.result,
                tool_call_id=function_call.call_id,
                tool_call_name=function_call.function.name,
                metrics={"time": t.elapsed},
            )
            if "tool_call_times" not in self.metrics:
                self.metrics["tool_call_times"] = {}
            if function_call.function.name not in self.metrics["tool_call_times"]:
                self.metrics["tool_call_times"][function_call.function.name] = []
            self.metrics["tool_call_times"][function_call.function.name].append(t.elapsed)
            results.append(m)
            self.function_call_stack.append(function_call)

            # -*- Check function call limit
            if len(self.function_call_stack) >= self.function_call_limit:
                self.deactivate_function_calls()
                break  # Exit early if we reach the function call limit

        return results

    def get_system_prompt_from_llm(self) -> Optional[str]:
        return self.system_prompt

    def get_instructions_from_llm(self) -> Optional[List[str]]:
        return self.instructions