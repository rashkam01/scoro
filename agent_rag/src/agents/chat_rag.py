from datetime import timedelta

from pydantic import BaseModel
from restack_ai.agent import agent, import_functions, log

with import_functions():
    from src.functions.llm_chat import LlmChatInput, Message, llm_chat
    from src.functions.lookup_sales import lookup_sales


class MessageEvent(BaseModel):
    content: str


class EndEvent(BaseModel):
    end: bool


@agent.defn()
class AgentRag:
    def __init__(self) -> None:
        self.end = False
        self.messages = []

    @agent.event
    async def message(self, message: MessageEvent) -> list[Message]:
        log.info(f"Received message: {message.content}")

        sales_info = await agent.step(
            function=lookup_sales, start_to_close_timeout=timedelta(seconds=120)
        )

        system_content = f"You are a helpful assistant that can help with sales data. Here is the sales information: {sales_info}"

        self.messages.append(Message(role="user", content=message.content or ""))

        completion = await agent.step(
            function=llm_chat,
            function_input=LlmChatInput(
                messages=self.messages, system_content=system_content
            ),
            start_to_close_timeout=timedelta(seconds=120),
        )

        log.info(f"completion: {completion}")

        self.messages.append(
            Message(
                role="assistant", content=completion.choices[0].message.content or ""
            )
        )

        return self.messages

    @agent.event
    async def end(self) -> EndEvent:
        log.info("Received end")
        self.end = True
        return {"end": True}

    @agent.run
    async def run(self, function_input: dict) -> None:
        await agent.condition(lambda: self.end)
