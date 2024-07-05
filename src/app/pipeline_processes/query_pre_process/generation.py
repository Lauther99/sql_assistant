from src.components.memory.memory import Memory
from src.components.models.models_interfaces import Base_LLM
from src.app.rag.rag_utils import base_llm_generation
from src.app.pipeline_processes.query_pre_process.prompts import (
    get_generate_request_prompt,
    get_greeting_response_prompt,
    get_simple_filter_prompt,
)
from src.app.pipeline_processes.query_pre_process.retrievers import retrieve_classify_examples


def generate_request(model: Base_LLM, memory: Memory) -> str:
    input = get_generate_request_prompt(memory)

    res = base_llm_generation(model, input, "generate-request")
    return res


def generate_request_type(model: Base_LLM, user_request: str) -> str:
    results = retrieve_classify_examples(user_request)
    
    input = get_simple_filter_prompt(user_request, results)

    res = base_llm_generation(model, input, "request-type")
    return res


def generate_greeting_response_call(model: Base_LLM, memory: Memory) -> str:
    input = get_greeting_response_prompt(memory)

    res = base_llm_generation(model, input, "greeting-response")
    return res
