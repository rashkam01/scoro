from datetime import timedelta
from typing import Literal
from pydantic import BaseModel
from restack_ai.agent import NonRetryableError, agent, import_functions, log

with import_functions():
    from src.functions.evaluation_function import (
        ExecutionInput,
        ExecutionOutput,
        execute_node,
    )

class EndEvent(BaseModel):
    end: bool

@agent.defn()
class EvaluationAgent:
    """Evaluation agent for processing evaluation requests."""
    
    schema = {
        "initial_event": "evaluate",
        "events": {
            "evaluate": {
                "input": ExecutionInput,
                "description": "Evaluate input using specified parameters",
                "default_input": {
                    "prompt": "Sample student answer",
                    "category": [1.0],
                    "language": "EN",
                    "provider": "default",
                    "model": "default"
                }
            }
        }
    }

    def __init__(self) -> None:
        self.end = False
        self.result = None

    @agent.event
    async def evaluate(self, event_input: ExecutionInput) -> ExecutionOutput:
        log.info(f"Received evaluation request: {event_input}")
        
        try:
            result = await agent.step(
                function=execute_node,
                function_input=event_input,
                start_to_close_timeout=timedelta(seconds=120),
            )
            self.result = result

            
            result = await agent.step(
                function=execute_node,
                function_input=event_input,
                execution_result = result,
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



