from typing import Literal
import os
import json

from pydantic import BaseModel
from restack_ai.function import NonRetryableError, function, log
from openai import OpenAI


def raise_exception(message: str) -> None:
    log.error(message)
    raise NonRetryableError(message)

class ExecutionInput(BaseModel):
    prompt: str
    category: list[float]
    language: Literal["EN", "DE"]
    provider: Literal["openai", "anthropic", "mistral"]  # specify allowed providers
    model: Literal["gpt-4.1-mini", "claude-2", "mistral-large"]  # specify allowed models

class EvaluationStep(BaseModel):
    instruction: int
    rationale: str
    reasoning: str

class ExecutionOutput(BaseModel):
    steps: list[EvaluationStep]

async def evaluate_with_openai(input_data: ExecutionInput) -> ExecutionOutput:
    try:
        log.info("OpenAI evaluation started", input_data=input_data)

        if os.environ.get("RESTACK_API_KEY") is None:
            raise_exception("RESTACK_API_KEY is not set")

        client = OpenAI(
            base_url="https://ai.restack.io", 
            api_key=os.environ.get("RESTACK_API_KEY")
        )


        result = client.beta.chat.completions.parse(
            model=input_data.model,
            messages=[
                {"role": "system", "content": "You are a execution assistant responsible for generating step-by-step execution for evaluating specific feature nodes within a structured task."},
                {"role": "user", "content": input_data.prompt}
            ],
            response_format=ExecutionOutput,
            temperature=0.0,

        )

        # Extract the evaluation steps from the response
        response_content = result.choices[0].message.content
        log.info("response_content", response_content=response_content)

        # Step 1: Parse the JSON string into a Python dict
        response_content_dict = json.loads(response_content)

        # Step 2: Now safely iterate
        steps = [
            EvaluationStep(
                instruction=step["instruction"],
                rationale=step["rationale"],
                reasoning=step["reasoning"]
            )
            for step in response_content_dict["steps"]
        ]

        log.info("OpenAI evaluation completed", steps=steps)
        return ExecutionOutput(steps=steps)

    except Exception as e:
        error_message = f"OpenAI evaluation failed: {e}"
        raise NonRetryableError(error_message) from e

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
