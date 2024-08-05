from src.app.pipeline_processes.sql_pre_process.generation import generate_enhanced_request
from src.components.memory.memory_interfaces import HumanMessage, AIMessage
from src.app.pipeline_processes.query_pre_process.retrievers import (
    retrieve_classify_examples,
)
from src.components.models.models_interfaces import Base_LLM
from src.app.pipeline_processes.query_pre_process.generation import (
    generate_greeting_response_call,
    generate_request,
    generate_request_type,
)
from src.components.memory.memory import Memory
from src.components.collector.collector import AppDataCollector, LLMResponseCollector
from src.utils.utils import clean_symbols


def query_pre_process(
    llm: Base_LLM,
    memory: Memory,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
):
    # Generando el request basado en los mensajes en memoria
    output = generate_request(llm, llm_collector, collector, memory)
    user_request = output["user_intent"]
    current_slots = output["slots"]
    current_conversation_summary = output["summary"]
    
    collector.user_request = str(user_request).strip()
    collector.current_conversation_data.current_slots = str(current_slots).strip()
    
    output = generate_enhanced_request(llm, llm_collector, collector)
    user_request = output["response"]
    collector.modified_user_request = user_request

    # Clasificando el request --- Parte 1: Recuperando ejemplos de la db
    classify_examples = retrieve_classify_examples(user_request)

    # Clasificando el request --- Parte 2: Clasificando el request en simple/complex
    output = generate_request_type(llm, llm_collector, user_request, classify_examples)
    request_type = output["type"]

    collector.request_type = clean_symbols(str(request_type).strip())
    collector.current_conversation_data.current_conversation_summary = str(current_conversation_summary).strip()

    return collector


def simple_request_process(
    llm: Base_LLM,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
):
    output = generate_greeting_response_call(llm, llm_collector, collector)

    collector.ai_pre_response = str(output["message"]).strip()
    return collector
