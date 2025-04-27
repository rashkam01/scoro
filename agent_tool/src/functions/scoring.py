# src/functions/scoring.py

from pydantic import BaseModel
from restack_ai.function import NonRetryableError, function, log

from src.functions.evaluation_function import ExecutionOutput

class ScoringInput(BaseModel):
    evaluation: ExecutionOutput

class ScoringOutput(BaseModel):
    score: float
    feedback: str

@function.defn()
async def score_evaluation(function_input: ScoringInput) -> ScoringOutput:
    try:
        log.info("Scoring evaluation started", input=function_input)

        # Super simple scoring logic based on number of steps
        steps_count = len(function_input.evaluation.steps)

        # For example: more steps = higher score
        score = min(1.0, steps_count / 10.0)  # Max score capped at 1.0

        feedback = "Good job!" if score > 0.5 else "Needs improvement."

        log.info("Scoring evaluation completed", score=score, feedback=feedback)

        return ScoringOutput(score=score, feedback=feedback)
    except Exception as e:
        error_message = f"Scoring evaluation failed: {e}"
        raise NonRetryableError(error_message) from e
