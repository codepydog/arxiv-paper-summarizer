"""@prompt"""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Callable
from functools import update_wrapper
from types import MethodType
from typing import (
    Any,
    Awaitable,
    Generic,
    Iterable,
    Literal,
    ParamSpec,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    final,
    overload,
)

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from arxiv_paper_summarizer.message import (
    AIMessage,
    Message,
    OpenAIMessageType,
    SystemMessage,
    UserMessage,
)
from arxiv_paper_summarizer.utils import get_env_var, split_union_type

P = ParamSpec("P")
R = TypeVar("R", covariant=True)
MessageLikeType: TypeAlias = Union[str, tuple[str, str], Message[Any]]
LLM_Type: TypeAlias = Union[OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI, Any]
OpenAIModel: TypeAlias = Literal["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]


class AsyncPromptFunction(Protocol[P, R]):
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class PromptFunction(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class PromptDecorator(Protocol):

    @overload
    def __call__(self, func: Callable[P, Awaitable[R]]) -> AsyncPromptFunction[P, R]: ...  # type: ignore[overload-overlap]

    @overload
    def __call__(self, func: Callable[P, R]) -> PromptFunction[P, R]: ...


class BaseOpenAIPromptFunction(Generic[P, R]):

    def __init__(
        self,
        name: str,
        parameters: Sequence[inspect.Parameter],
        return_type: type[R],
        messages: Sequence[MessageLikeType],
        llm: LLM_Type | None,
        model_name: OpenAIModel | None = None,
        response_format: Any | None = None,
    ) -> None:
        self._name = name
        self._llm = llm
        self._signature = inspect.Signature(
            parameters=parameters,
            return_annotation=return_type,
        )
        self._messages = messages
        self._model_name = model_name
        self._return_types = split_union_type(return_type)
        self._response_format = response_format

    @property
    def return_types(self) -> Sequence[type[R]]:
        return self._return_types

    @property
    def model_name(self) -> str:
        return self._model_name or self.get_model_name()

    @property
    def llm(self) -> LLM_Type:
        return self._llm or self.get_llm()

    @property
    def response_format(self) -> Any | None:
        return self._response_format

    def get_model_name(self) -> str:
        return "gpt-4o"

    def get_llm(self) -> LLM_Type:
        return NotImplemented

    def format(self, *args: P.args, **kwargs: P.kwargs) -> Iterable[OpenAIMessageType]:
        """Format the messages with the given arguments."""
        bound_args = self.get_bound_args(*args, **kwargs)
        formatted_messages = []
        for message in self._messages:
            msg_obj: Message[Any] = self._format(message)
            formatted_message = msg_obj.format(**bound_args)
            formatted_messages.append(formatted_message)
        return cast(Iterable[OpenAIMessageType], formatted_messages)

    def _format(self, message: MessageLikeType) -> Message[Any]:
        if isinstance(message, str):
            return UserMessage(message)

        if isinstance(message, tuple) and len(message) == 2 and isinstance(message[0], str) and isinstance(message[1], str):
            role, text = message[0].lower(), message[1]
            role_map = {"user": UserMessage, "system": SystemMessage, "ai": AIMessage}
            if role_cls := role_map.get(role):
                return role_cls(text)
            raise ValueError(f"Invalid role: '{role}' in message: {message}")

        if isinstance(message, Message):
            return message

        raise TypeError(f"Invalid message type: {type(message)}. Expected str, tuple, or Message instance.")

    def get_bound_args(self, *args: P.args, **kwargs: P.kwargs) -> OrderedDict[str, Any]:
        """Get the bound arguments for the function."""
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        bound_args.arguments.pop("self", None)  # Avoid passing `self` twice
        return bound_args.arguments

    def parse_completion_content(self, completion: ChatCompletion) -> R:
        return cast(R, completion.choices[0].message.content)

    def __call__(self, *args: P.args, **kwargs: P.kwargs):
        """OpenAI prompt function."""
        return NotImplemented

    def __get__(self, instance: object, owner: object) -> MethodType:
        return MethodType(self, instance)


class AsyncOpenAIPromptFunction(BaseOpenAIPromptFunction[P, R], Generic[P, R], AsyncPromptFunction[P, R]):
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Async OpenAI prompt function."""
        messages = self.format(*args, **kwargs)
        if self.response_format:
            chat_completion = await cast(
                ChatCompletion,
                self.llm.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=self.response_format,
                ),
            )  # type: ignore[misc]
            message = chat_completion.choices[0].message
            if getattr(message, "refusal", None):
                return cast(R, message.refusal)
            return cast(R, message.parsed)  # type: ignore[attr-defined]

        chat_completion = await self.llm.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )  # type: ignore[misc]
        return cast(R, self.parse_completion_content(chat_completion))

    @final
    def get_llm(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=get_env_var("OPENAI_API_KEY"),
            base_url=get_env_var("OPENAI_BASE_URL"),
        )


class OpenAIPromptFunction(BaseOpenAIPromptFunction[P, R], Generic[P, R], PromptFunction[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """OpenAI prompt function."""
        messages = self.format(*args, **kwargs)
        if self.response_format:
            chat_completion = cast(
                ChatCompletion,
                self.llm.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=self.response_format,
                ),
            )
            message = chat_completion.choices[0].message
            if message.refusal:
                return cast(R, message.refusal)
            return cast(R, message.parsed)  # type: ignore[attr-defined]

        chat_completion = cast(
            ChatCompletion,
            self.llm.chat.completions.create(
                model=self.model_name,
                messages=messages,
            ),
        )
        return cast(R, self.parse_completion_content(chat_completion))

    @final
    def get_llm(self) -> OpenAI:
        return OpenAI(
            api_key=get_env_var("OPENAI_API_KEY"),
            base_url=get_env_var("OPENAI_BASE_URL"),
        )


def openai_prompt(
    *messages: MessageLikeType,
    llm: LLM_Type | None = None,
    model_name: OpenAIModel | None = None,
    response_format: Any | None = None,
) -> PromptDecorator:
    def decorator(
        func: Callable[P, Awaitable[R]] | Callable[P, R],
    ) -> AsyncOpenAIPromptFunction[P, R] | OpenAIPromptFunction[P, R]:
        func_signature = inspect.signature(func)
        if inspect.iscoroutinefunction(func):
            async_prompt_func = AsyncOpenAIPromptFunction[P, R](
                name=func.__name__,
                parameters=list(func_signature.parameters.values()),
                return_type=func_signature.return_annotation,
                messages=messages,
                llm=llm,
                model_name=model_name,
                response_format=response_format,
            )
            return cast(AsyncOpenAIPromptFunction[P, R], update_wrapper(async_prompt_func, func))

        prompt_func = OpenAIPromptFunction[P, R](
            name=func.__name__,
            parameters=list(func_signature.parameters.values()),
            return_type=func_signature.return_annotation,
            messages=messages,
            llm=llm,
            model_name=model_name,
            response_format=response_format,
        )
        return cast(OpenAIPromptFunction[P, R], update_wrapper(prompt_func, func))

    return decorator  # type: ignore[return-value]
