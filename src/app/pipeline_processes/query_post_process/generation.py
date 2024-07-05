from src.components.models.models_interfaces import Base_LLM
from src.app.rag.rag_utils import base_llm_generation
from src.app.pipeline_processes.query_post_process.prompts import translator_template


def translate_response(model: Base_LLM, user_input, actual_answer):
    input = translator_template.format(
        user_input=user_input, actual_answer=actual_answer
    )
    res = base_llm_generation(model, input, "post-process-translation")
    return res
