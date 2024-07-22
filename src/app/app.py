from typing import Any, Hashable
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
from src.components.memory.memory_interfaces import AIMessage
from src.components.models.embeddings.embeddings import (
    HF_MultilingualE5_Embeddings,
    Openai_Embeddings,
)
from src.components.models.llms.llms import HF_Llama38b_LLM, Openai_LLM


def main_pipeline(
    openai_llm: Openai_LLM,
    hf_llm: HF_Llama38b_LLM,
    mle5_embeddings: HF_MultilingualE5_Embeddings,
    openai_embeddings: Openai_Embeddings,
    memory: Memory,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
):
    _pre_process_pipeline(
        llm=hf_llm, memory=memory, collector=collector, llm_collector=llm_collector
    )
    is_simple = collector.request_type.lower().strip() == "simple"

    if is_simple:
        _simple_request_pipeline(
            llm=hf_llm, collector=collector, llm_collector=llm_collector
        )
    else:
        _complex_request_pipeline(
            hf_llm=hf_llm,
            openai_llm=openai_llm,
            mle5_embeddings=mle5_embeddings,
            openai_embeddings=openai_embeddings,
            collector=collector,
            llm_collector=llm_collector,
            memory=memory,
        )
        _post_sql_generation_pipeline(openai_llm, collector, llm_collector)

    _post_process_pipeline(openai_llm, collector=collector, llm_collector=llm_collector)


def chat(
    user_message: str,
    ai_response: AIMessage | None,
    chat_config: ChatConfig,
    current_conversation_summary: str = None,
    current_conversation_slots: str = None,
):
    memory = chat_config.memory
    collector = chat_config.collector
    llm_collector = chat_config.llm_collector
    openai_llm: Openai_LLM = chat_config.openai_llm
    hf_llm: HF_Llama38b_LLM = chat_config.hf_llm
    mle5_embeddings: HF_MultilingualE5_Embeddings = chat_config.mle5_embeddings
    openai_embeddings: Openai_Embeddings = chat_config.openai_embeddings

    # Actualizamos el collector con datos de la conversacion
    last_user_message = memory.add_user_message(user_message)
    collector.current_conversation_data.last_user_message = last_user_message
    collector.current_conversation_data.current_conversation_summary = (
        current_conversation_summary
    )
    collector.current_conversation_data.current_slots = current_conversation_slots
    collector.current_conversation_data.last_ai_message = ai_response

    main_pipeline(
        openai_llm=openai_llm,
        hf_llm=hf_llm,
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

    return collector
