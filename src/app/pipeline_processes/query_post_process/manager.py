from src.app.pipeline_processes.query_post_process.generation import translate_response
from src.components.models.models_interfaces import Base_LLM
from src.components.collector.collector import AppDataCollector, LLMResponseCollector


def query_post_process(llm: Base_LLM, collector: AppDataCollector, llm_collector: LLMResponseCollector):
    actual_answer = collector.ai_pre_response
    user_message = collector.current_conversation_data.last_user_message.message
    
    response = translate_response(llm, llm_collector, user_message, actual_answer)
    
    collector.ai_post_response = response["response"]
    collector.detected_language = response["detected_language"]
    return response