from src.components.memory.memory import Memory
from src.components.models.models_interfaces import Base_LLM
from src.app.rag.rag_utils import base_llm_generation
from src.app.pipeline_processes.query_pre_process.prompts import (
    get_greeting_response_prompt,
    get_simple_filter_prompt,
)
from src.components.collector.collector import LLMResponseCollector, AppDataCollector
from src.components.memory.memory_interfaces import HumanMessage, AIMessage


def generate_request(
    model: Base_LLM,
    llm_collector: LLMResponseCollector,
    collector: AppDataCollector,
    memory: Memory,
):
    """
    Genera una solicitud utilizando un modelo de lenguaje y un contexto de memoria.

    Esta función crea un prompt basado en el contexto de memoria proporcionado y aplica una plantilla
    de modelo para generar una solicitud. El resultado es un diccionario que contiene una intención.

    ### Args:
    - `model (Base_LLM)`: El modelo de lenguaje que se utilizará para generar la solicitud.
    - `memory (Memory)`: El contexto de memoria que proporciona la información necesaria para generar la solicitud.

    ### Returns:
    - `dict`: Un diccionario con una única clave `'intention'`, que tiene un valor de cadena representando la intención de la solicitud.

    ### Ejemplo de respuesta:
    ```json
    {
        "intention": "The human is requesting information from the EMED-2012-LR-P12 measurement system."
    }
    ```
    """
    user_message = collector.current_conversation_data.last_user_message
    ai_message = collector.current_conversation_data.last_ai_message
    dictionary = collector.terms_dictionary
    current_slots = collector.current_conversation_data.current_slots
    current_conversation_summary = (
        collector.current_conversation_data.current_conversation_summary
    )

    instruction, suffix = memory.get_new_summary_instruction(
        user_message=user_message,
        ai_message=ai_message,
        current_slots=current_slots,
        current_summary=current_conversation_summary,
        dictionary=dictionary
    )
    
    prompt = model.apply_model_template(instruction, suffix)
    res = base_llm_generation(model, llm_collector, prompt, "generate-request")
    return res


def generate_request_type(
    model: Base_LLM,
    llm_collector: LLMResponseCollector,
    user_request: str,
    classify_examples: tuple,
):
    """
    Genera un tipo de solicitud utilizando un modelo de lenguaje y ejemplos de clasificación.

    Esta función crea un prompt basado en la solicitud del usuario y ejemplos de clasificación,
    y aplica una plantilla de modelo para generar el tipo de solicitud. El resultado es una cadena
    que describe el tipo de solicitud.

    ### Args:
    - `model (Base_LLM)`: El modelo de lenguaje que se utilizará para generar el tipo de solicitud.
    - `user_request (str)`: La solicitud del usuario que se va a clasificar.
    - `classify_examples (tuple)`: Ejemplos utilizados para ayudar a clasificar la solicitud.

    ### Returns:
    - `str`: Una cadena que describe el tipo de solicitud.

    ### Ejemplo de respuesta:
    ```json
    {
        "analysis": "This input is out of my knowledge, this type is complex",
        "type": "complex"
    }
    ```
    """
    instruction, suffix = get_simple_filter_prompt(user_request, classify_examples)
    prompt = model.apply_model_template(instruction, suffix)
    res = base_llm_generation(model, llm_collector, prompt, "request-type")
    return res


def generate_greeting_response_call(
    model: Base_LLM, llm_collector: LLMResponseCollector, collector: AppDataCollector,
):
    last_user_message = collector.current_conversation_data.last_user_message
    last_ai_message = collector.current_conversation_data.last_ai_message
    
    instruction, suffix = get_greeting_response_prompt(last_user_message, last_ai_message)
    prompt = model.apply_model_template(instruction, suffix)
    res = base_llm_generation(model, llm_collector, prompt, "greeting-response")
    return res
