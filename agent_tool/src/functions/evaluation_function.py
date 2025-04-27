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
    prompt_execution: str
    prompt_scoring: str
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

class ScoringOutput(BaseModel):
    value: float
    rationale: str
    feedback: str

class ScoringInput(BaseModel):
    """Combined input for scoring that includes both the execution input and result."""
    execution_input: ExecutionInput
    execution_result: ExecutionOutput

async def execute_with_openai(input_data: ExecutionInput) -> ExecutionOutput:
    """Execute evaluation with OpenAI."""
    try:
        log.info("OpenAI execution started", input_data=input_data)
        prompt = input_data.prompt_execution

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
                {"role": "user", "content": prompt}
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

        log.info("OpenAI execution completed", steps=steps)
        return ExecutionOutput(steps=steps)

    except Exception as e:
        error_message = f"OpenAI execution failed: {e}"
        raise NonRetryableError(error_message) from e

async def score_with_openai(input_data: ExecutionInput, execution_result: str) -> ScoringOutput:
    """Score evaluation with OpenAI."""
    try:
        prompt = input_data.prompt_scoring + execution_result
        log.info("Prompt Score", prompt=prompt)


        if os.environ.get("RESTACK_API_KEY") is None:
            raise_exception("RESTACK_API_KEY is not set")

        client = OpenAI(
            base_url="https://ai.restack.io",
            api_key=os.environ.get("RESTACK_API_KEY")
        )

        result = client.beta.chat.completions.parse(
            model=input_data.model,
            messages=[
                {"role": "system", "content": "You are a scoring assistant responsible for evaluating execution results."},
                {"role": "user", "content": prompt}
            ],
            response_format=ScoringOutput,
            temperature=0.0,
        )

        # Extract the scoring result from the response
        response_content = result.choices[0].message.content
        log.info("response_content", response_content=response_content)

        # Parse the JSON string into a Python dict
        response_content_dict = json.loads(response_content)

        # Create and return the ScoringOutput
        scoring_output = ScoringOutput(**response_content_dict)
        log.info("OpenAI scoring completed", scoring_output=scoring_output)
        return scoring_output

    except Exception as e:
        error_message = f"OpenAI scoring failed: {e}"
        raise NonRetryableError(error_message) from e

async def execute_with_anthropic(input_data: ExecutionInput) -> ExecutionOutput:
    """Execute evaluation with Anthropic."""
    # Anthropic-specific execution logic
    steps = [
        EvaluationStep(
            instruction=1,
            rationale="Anthropic evaluation",
            reasoning=f"Using model: {input_data.model}"
        )
    ]
    return ExecutionOutput(steps=steps)

async def score_with_anthropic(input_data: ExecutionInput, execution_result: str) -> ScoringOutput:
    """Score evaluation with Anthropic."""
    # Anthropic-specific scoring logic
    return ScoringOutput(
        value=0.5,  # Default value
        rationale=f"Anthropic scoring with model: {input_data.model}",
        feedback="This is a placeholder for Anthropic scoring"
    )

async def execute_with_mistral(input_data: ExecutionInput) -> ExecutionOutput:
    """Execute evaluation with Mistral."""
    # Mistral-specific execution logic
    steps = [
        EvaluationStep(
            instruction=1,
            rationale="Mistral evaluation",
            reasoning=f"Using model: {input_data.model}"
        )
    ]
    return ExecutionOutput(steps=steps)

async def score_with_mistral(input_data: ExecutionInput, execution_result: str) -> ScoringOutput:
    """Score evaluation with Mistral."""
    # Mistral-specific scoring logic
    return ScoringOutput(
        value=0.5,  # Default value
        rationale=f"Mistral scoring with model: {input_data.model}",
        feedback="This is a placeholder for Mistral scoring"
    )

@function.defn()
async def execute_node(function_input: ExecutionInput) -> ExecutionOutput:
    try:
        log.info("execute_node function started", function_input=function_input)

        if isinstance(function_input, dict):
            function_input = ExecutionInput(**function_input)

        # Route to appropriate execution function based on provider
        provider_map = {
            "openai": execute_with_openai,
            "anthropic": execute_with_anthropic,
            "mistral": execute_with_mistral
        }

        if function_input.provider not in provider_map:
            raise NonRetryableError(f"Unsupported provider: {function_input.provider}")

        execution_function = provider_map[function_input.provider]
        result = await execution_function(function_input)

        return result
    except Exception as e:
        error_message = f"Evaluation failed: {e}"
        raise NonRetryableError(error_message) from e


@function.defn()
async def score_node(function_input: ScoringInput) -> ScoringOutput:
    try:
        log.info("score_node function started")
        log.info(f"function_input type: {type(function_input)}")
        log.info(f"function_input: {function_input}")

        # Handle different input formats
        execution_input = None
        execution_result = None

        if isinstance(function_input, ScoringInput):
            # If we get a proper ScoringInput object
            log.info("Received ScoringInput object")
            execution_input = function_input.execution_input
            execution_result = function_input.execution_result
        elif isinstance(function_input, dict):
            # If we get a dict, try to extract the components
            log.info("Received dict, extracting components")
            if "execution_input" in function_input and "execution_result" in function_input:
                execution_input = ExecutionInput(**function_input["execution_input"])
                execution_result = ExecutionOutput(**function_input["execution_result"])
            else:
                # Assume the dict itself is the execution_input and look for execution_result elsewhere
                log.info("Dict doesn't contain expected keys, using as execution_input")
                execution_input = ExecutionInput(**function_input)
                # execution_result will be None, handled below
        else:
            # Unexpected input type
            error_msg = f"Unexpected input type: {type(function_input)}"
            log.error(error_msg)
            raise NonRetryableError(error_msg)

        log.info(f"Extracted execution_input: {execution_input}")
        log.info(f"Extracted execution_result: {execution_result}")

        if execution_result is None:
            error_msg = "Missing execution_result"
            log.error(error_msg)
            raise NonRetryableError(error_msg)

        # Route to appropriate scoring function based on provider
        provider_map = {
            "openai": score_with_openai,
            "anthropic": score_with_anthropic,
            "mistral": score_with_mistral
        }

        log.info(f"Provider: {execution_input.provider}")
        if execution_input.provider not in provider_map:
            error_msg = f"Unsupported provider: {execution_input.provider}"
            log.error(error_msg)
            raise NonRetryableError(error_msg)

        try:
            execution_output_str = json.dumps(execution_result.dict(), indent=2)
            log.info("execution_str created successfully")
        except Exception as e:
            log.error(f"Error converting execution_result to string: {e}")
            # Try a fallback approach
            try:
                if hasattr(execution_result, "__dict__"):
                    execution_output_str = json.dumps(execution_result.__dict__, indent=2)
                    log.info("Used __dict__ fallback for execution_result")
                else:
                    execution_output_str = str(execution_result)
                    log.info("Used str() fallback for execution_result")
            except Exception as e2:
                log.error(f"Fallback also failed: {e2}")
                execution_output_str = "{}"  # Empty JSON as last resort
                log.info("Using empty JSON as last resort")

        log.info(f"execution_output_str: {execution_output_str}")

        scoring_function = provider_map[execution_input.provider]
        log.info(f"Selected scoring function: {scoring_function.__name__}")

        result = await scoring_function(execution_input, execution_output_str)
        log.info(f"Scoring function returned result: {result}")

        return result
    except Exception as e:
        error_message = f"Evaluation failed: {e}"
        raise NonRetryableError(error_message) from e
