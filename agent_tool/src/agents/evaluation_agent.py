from datetime import timedelta
from pydantic import BaseModel
from restack_ai.agent import NonRetryableError, agent, import_functions, log

with import_functions():
    from src.functions.evaluation_function import (
        ExecutionInput,
        ExecutionOutput,
        evaluate_answer,
    )

class EndEvent(BaseModel):
    end: bool

@agent.defn()
class EvaluationAgent:
    def __init__(self) -> None:
        self.end = False
        self.result = None

    @agent.event
    async def evaluate(self, event_input: ExecutionInput) -> ExecutionOutput:
        log.info(f"Received evaluation request: {event_input}")
        
        try:
            result = await agent.step(
                function=evaluate_answer,
                function_input=event_input,  # We can pass event_input directly since it's ExecutionInput
                start_to_close_timeout=timedelta(seconds=120),
            )
            self.result = result
            return result
        except Exception as e:
            error_message = f"Error during evaluation: {e}"
            raise NonRetryableError(error_message) from e

    @agent.event
    async def end(self) -> EndEvent:
        log.info("Received end")
        self.end = True
        return EndEvent(end=True)

    @agent.run
    async def run(self, agent_input: dict) -> None:
        log.info("EvaluationAgent agent_input", agent_input=agent_input)
        await agent.condition(lambda: self.end)

