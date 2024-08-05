from datetime import datetime
from typing import Any, Hashable, List, Union
from src.db.mongo.interfaces import ChatDocument
from src.app.interfaces import ChatConfig
from src.app.pipeline_processes.pipelines import (
    _complex_request_pipeline,
    _post_process_pipeline,
    _post_sql_generation_pipeline,
    _pre_process_pipeline,
    _simple_request_pipeline,
)
from src.components.collector.collector import AppDataCollector, LLMResponseCollector
from src.components.memory.memory import Memory
from src.components.memory.memory_interfaces import AIMessage, HumanMessage
from src.components.models.embeddings.embeddings import (
    HF_MultilingualE5_Embeddings,
    Openai_Embeddings,
)
from src.components.models.llms.llms import HF_Llama38b_LLM, Openai_LLM


def main_pipeline(
    openai_llm: Openai_LLM,
    llama3_llm: HF_Llama38b_LLM,
    mle5_embeddings: HF_MultilingualE5_Embeddings,
    openai_embeddings: Openai_Embeddings,
    memory: Memory,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
):
    _pre_process_pipeline(
        llm=llama3_llm, memory=memory, collector=collector, llm_collector=llm_collector
    )
    is_simple = collector.request_type.lower().strip() == "simple"

    if is_simple:
        _simple_request_pipeline(
            llm=llama3_llm, collector=collector, llm_collector=llm_collector
        )
    else:
        _complex_request_pipeline(
            llama3_llm=llama3_llm,
            openai_llm=openai_llm,
            mle5_embeddings=mle5_embeddings,
            openai_embeddings=openai_embeddings,
            collector=collector,
            llm_collector=llm_collector,
            memory=memory,
        )
        _post_sql_generation_pipeline(llama3_llm, collector, llm_collector)

    _post_process_pipeline(openai_llm, collector=collector, llm_collector=llm_collector)


def chat(
    new_user_message: str,
    chat_document: ChatDocument,
):
    
    # Modelos LLM
    llama3_llm = HF_Llama38b_LLM()
    openai_llm = Openai_LLM()
    
    # Modelos Embeddings
    mle5_embeddings = HF_MultilingualE5_Embeddings()
    openai_embeddings = Openai_Embeddings()
    
    #Iniciando los modelos
    openai_llm.init_model()
    llama3_llm.init_model()
    mle5_embeddings.init_model()
    openai_embeddings.init_model()
    
    # Collectors
    collector = AppDataCollector()
    llm_collector = LLMResponseCollector()
    
    # Memoria
    memory = Memory(chat_document)
    
    # Actualizamos el collector con datos de la conversacion
    last_user_message = memory.add_user_message(new_user_message)
    collector.current_conversation_data.last_user_message = last_user_message
    collector.current_conversation_data.current_conversation_summary = (
        chat_document.current_summary
    )
    collector.current_conversation_data.current_slots = chat_document.conversation_slots
    collector.current_conversation_data.last_ai_message = chat_document.last_assistant_message

    main_pipeline(
        openai_llm=openai_llm,
        llama3_llm=llama3_llm,
        mle5_embeddings=mle5_embeddings,
        openai_embeddings=openai_embeddings,
        memory=memory,
        collector=collector,
        llm_collector=llm_collector,
    )

    ai_message = collector.ai_post_response
    dataframe: list[dict[Hashable, Any]] | None = collector.dataframe_response
    sql_response = (
        collector.sql_code
        if collector.sql_pre_query is None
        else collector.sql_pre_query
    )

    last_ai_message = memory.add_ai_message(
        ai_message, last_user_message.message_id, dataframe, sql_response
    )
    collector.current_conversation_data.last_ai_message = last_ai_message

    chat_document.last_user_message = collector.current_conversation_data.last_user_message
    chat_document.last_assistant_message = collector.current_conversation_data.last_ai_message
    chat_document.last_interaction = datetime.now()
    
    if len(chat_document.messages) >= 8:
        chat_document.conversation_slots = collector.current_conversation_data.current_slots
        chat_document.current_summary = collector.current_conversation_data.current_conversation_summary
    else:
        chat_document.conversation_slots = None
        chat_document.current_summary = None
        

    return llm_collector, collector, chat_document
