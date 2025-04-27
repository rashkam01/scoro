from typing import Literal
from pydantic import BaseModel
from restack_ai.function import NonRetryableError, function, log

class ExecutionInput(BaseModel):
    prompt: str
    category: list[float]
    language: Literal["EN", "DE"]
    provider: Literal["openai", "anthropic", "mistral"]  # specify allowed providers
    model: Literal["gpt-4", "claude-2", "mistral-large"]  # specify allowed models

class EvaluationStep(BaseModel):
    instruction: int
    rationale: str
    reasoning: str

class ExecutionOutput(BaseModel):
    steps: list[EvaluationStep]

async def evaluate_with_openai(input_data: ExecutionInput) -> ExecutionOutput:
    # OpenAI-specific evaluation logic
    steps = [
        EvaluationStep(
            instruction=1,
            rationale="OpenAI evaluation",
            reasoning=f"Using model: {input_data.model}"
        )
    ]
    return ExecutionOutput(steps=steps)

async def evaluate_with_anthropic(input_data: ExecutionInput) -> ExecutionOutput:
    # Anthropic-specific evaluation logic
    steps = [
        EvaluationStep(
            instruction=1,
            rationale="Anthropic evaluation",
            reasoning=f"Using model: {input_data.model}"
        )
    ]
    return ExecutionOutput(steps=steps)

async def evaluate_with_mistral(input_data: ExecutionInput) -> ExecutionOutput:
    # Mistral-specific evaluation logic
    steps = [
        EvaluationStep(
            instruction=1,
            rationale="Mistral evaluation",
            reasoning=f"Using model: {input_data.model}"
        )
    ]
    return ExecutionOutput(steps=steps)

@function.defn()
async def execute_node(function_input: ExecutionInput) -> ExecutionOutput:
    try:
        log.info("evaluate_answer function started", function_input=function_input)
        
        # Route to appropriate evaluation function based on provider
        provider_map = {
            "openai": evaluate_with_openai,
            "anthropic": evaluate_with_anthropic,
            "mistral": evaluate_with_mistral
        }
        
        if function_input.provider not in provider_map:
            raise NonRetryableError(f"Unsupported provider: {function_input.provider}")
            
        evaluation_function = provider_map[function_input.provider]
        result = await evaluation_function(function_input)
        
        return result
    except Exception as e:
        error_message = f"Evaluation failed: {e}"
        raise NonRetryableError(error_message) from e
