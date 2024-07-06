from src.components.models.models_interfaces import Base_LLM
from src.app.rag.rag_utils import base_llm_generation
from src.app.pipeline_processes.query_post_process.prompts import get_translator_prompt


def translate_response(model: Base_LLM, user_input, actual_answer):
    instruction, suffix = get_translator_prompt(user_input, actual_answer)
    
    prompt = model.apply_model_template(instruction, suffix)
    
    res = base_llm_generation(model, prompt, "post-process-translation")
    return res
