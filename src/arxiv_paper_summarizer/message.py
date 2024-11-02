"""Message."""

import string
from typing import ClassVar, Generic, ParamSpec, TypedDict, TypeVar, Union

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

P = ParamSpec("P")
R = TypeVar("R")

OpenAIMessageType = Union[
    ChatCompletionSystemMessageParam
    | ChatCompletionUserMessageParam
    | ChatCompletionAssistantMessageParam
    | ChatCompletionToolMessageParam
    | ChatCompletionFunctionMessageParam
]


class MessageDict(TypedDict):
    role: str
    content: str


class Message(Generic[R]):
    role: ClassVar[str] = "user"

    def __init__(self, content: str):
        self.content = content

    def format(self, *args: P.args, **kwargs: P.kwargs) -> MessageDict:
        formatter = string.Formatter()
        field_names = {field_name for _, field_name, _, _ in formatter.parse(self.content) if field_name is not None and field_name != ""}
        relevant_kwargs = {key: kwargs[key] for key in field_names if key in kwargs}
        missing_fields = field_names - relevant_kwargs.keys()
        if missing_fields:
            missing = ", ".join(f"'{field}'" for field in missing_fields)
            raise KeyError(f"Missing arguments for message formatting: {missing}")

        content = self.content.format(**relevant_kwargs)
        return {"role": self.role, "content": content}


class UserMessage(Message[R]):
    role = "user"


class SystemMessage(Message[R]):
    role = "system"


class AIMessage(Message[R]):
    role = "ai"


class FunctionMessage(Message[R]):
    role = "function"
