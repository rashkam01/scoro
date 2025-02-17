import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from restack_ai.function import FunctionFailure, function, log

load_dotenv()


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LlmChatInput(BaseModel):
    system_content: str | None = None
    model: str | None = None
    messages: list[Message] | None = None


def raise_exception(message: str) -> None:
    log.error(message)
    raise FunctionFailure(message, non_retryable=True)


@function.defn()
async def llm_chat(agent_input: LlmChatInput) -> ChatCompletion:
    try:
        log.info("llm_chat function started", agent_input=agent_input)

        if os.environ.get("RESTACK_API_KEY") is None:
            error_message = "RESTACK_API_KEY is not set"
            raise_exception(error_message)

        client = OpenAI(
            base_url="https://ai.restack.io", api_key=os.environ.get("RESTACK_API_KEY")
        )

        if agent_input.system_content:
            agent_input.messages.append(
                {"role": "system", "content": agent_input.system_content}
            )

        response = client.chat.completions.create(
            model=agent_input.model or "gpt-4o-mini",
            messages=agent_input.messages,
        )
    except Exception as e:
        log.error("llm_chat function failed", error=e)
        raise
    else:
        log.info("llm_chat function completed", response=response)

        return response
