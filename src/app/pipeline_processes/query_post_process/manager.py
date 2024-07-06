from src.app.pipeline_processes.query_post_process.generation import translate_response
from src.components.models.models_interfaces import Base_LLM


def query_post_process(model: Base_LLM, user_input: str, actual_answer: str):
    response = translate_response(model, user_input, actual_answer)
    return response
