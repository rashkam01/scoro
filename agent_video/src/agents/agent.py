from datetime import timedelta

from pydantic import BaseModel
from restack_ai.agent import agent, import_functions, log

with import_functions():
    from src.functions.llm_chat import LlmChatInput, Message, llm_chat
    from src.functions.context_docs import context_docs

class MessagesEvent(BaseModel):
    messages: list[Message]


class EndEvent(BaseModel):
    end: bool


@agent.defn()
class AgentVideo:
    def __init__(self) -> None:
        self.end = False
        self.messages: list[Message] = []

    @agent.event
    async def messages(self, messages_event: MessagesEvent) -> list[Message]:
        log.info(f"Received message: {messages_event.messages}")
        self.messages.extend(messages_event.messages)

        assistant_message = await agent.step(
            function=llm_chat,
            function_input=LlmChatInput(messages=self.messages),
            start_to_close_timeout=timedelta(seconds=120),
        )
        self.messages.append(Message(role="assistant", content=str(assistant_message)))
        return self.messages

    @agent.event
    async def end(self, end: EndEvent) -> EndEvent:
        log.info("Received end")
        self.end = True
        return end

    @agent.run
    async def run(self) -> None:
        docs = await agent.step(function=context_docs)
        system_prompt=f"""
        You are an interactive video assistant, your answers will be used in text to speech so try to keep answers short and concise so that interaction is seamless.
        You can answer questions about the following documentation:
        {docs}
        """
        self.messages.append(Message(role="system", content=system_prompt))
        await agent.condition(lambda: self.end)
