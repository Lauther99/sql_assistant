from src.app.pipeline_processes.query_pre_process.retrievers import retrieve_classify_examples
from src.components.models.models_interfaces import Base_LLM
from src.app.pipeline_processes.query_pre_process.generation import (
    generate_greeting_response_call,
    generate_request,
    generate_request_type,
)
from src.components.memory.memory import Memory
from src.app.app import AppDataCollector



def query_pre_process(llm: Base_LLM, memory: Memory, collector: AppDataCollector):
    # Generando el request basado en los mensajes en memoria
    output = generate_request(llm, memory)
    user_request = output["intention"]
    
    # Clasificando el request --- Parte 1: Recuperando ejemplos de la db
    classify_examples = retrieve_classify_examples(user_request)
    
    # Clasificando el request --- Parte 2: Clasificando el request en simple/complex
    output = generate_request_type(llm, user_request, classify_examples)
    request_type = output["type"]
    
    collector.user_request = user_request
    collector.request_type = request_type
    
    return collector


def simple_request_process(llm: Base_LLM, memory: Memory, collector: AppDataCollector):
    output = generate_greeting_response_call(llm, memory)
    
    collector.simple_response = output["message"]
    return collector