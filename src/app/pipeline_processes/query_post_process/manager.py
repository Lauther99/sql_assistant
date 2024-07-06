from src.app.pipeline_processes.query_post_process.generation import translate_response
from src.components.models.models_interfaces import Base_LLM
from src.app.app import AppDataCollector


def query_post_process(llm: Base_LLM, collector: AppDataCollector, user_message: str):
    actual_answer = collector.ai_pre_response
    
    response = translate_response(llm, user_message, actual_answer)
    
    collector.ai_post_response = response["response"]
    collector.detected_language = response["detected_language"]
    return response