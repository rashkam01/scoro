from typing import Literal
from pydantic import BaseModel
from restack_ai.function import NonRetryableError, function, log

class ExecutionInput(BaseModel):
    prompt: str
    category: list[float]
    language: Literal["EN", "DE"]
    provider: str
    model: str

class EvaluationStep(BaseModel):
    instruction: int
    rationale: str
    reasoning: str

class ExecutionOutput(BaseModel):
    steps: list[EvaluationStep]

@function.defn()
async def evaluate_answer(function_input: ExecutionInput) -> ExecutionOutput:
    try:
        log.info("evaluate_answer function started", function_input=function_input)
        
        # This is a simplified example - you would typically integrate with your 
        # evaluation logic here
        steps = [
            EvaluationStep(
                instruction=1,
                rationale="Initial comparison of answers",
                reasoning="Comparing student answer with sample solution"
            ),
            EvaluationStep(
                instruction=2,
                rationale="Verification of answer correctness",
                reasoning="Checking if the answer matches expected format"
            ),
            EvaluationStep(
                instruction=3,
                rationale="Application of scoring criteria",
                reasoning="Evaluating based on provided category weights"
            ),
            EvaluationStep(
                instruction=4,
                rationale="Final assessment",
                reasoning="Determining overall evaluation result"
            )
        ]
        
        return ExecutionOutput(steps=steps)
    except Exception as e:
        error_message = f"Evaluation failed: {e}"
        raise NonRetryableError(error_message) from e