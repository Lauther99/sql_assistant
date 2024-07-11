from src.components.models.models_interfaces import Base_LLM
from src.components.memory.memory import Memory
from src.app.pipeline_processes.sql_pre_process.retrievers import (
    retrieve_semantic_term_definitions,
    retrieve_sql_semantic_information,
    retrieve_terms_examples,
)
from src.app.pipeline_processes.query_pre_process.manager import query_pre_process
from src.app.pipeline_processes.sql_pre_process.generation import (
    generate_chat_summary,
    generate_flavored_request,
    generate_multi_definition_detector,
    generate_multi_definition_question,
    generate_request_from_chat_summary,
    generate_semantic_info,
    generate_technical_terms,
)
from src.components.models.llms.llms import HF_Llama38b_LLM, Langchain_OpenAI_LLM
from src.components.models.embeddings.embeddings import (
    HF_MultilingualE5_Embeddings,
    Openai_Embeddings,
)
from src.components.collector.collector import AppDataCollector, LLMResponseCollector


def complex_request_process_modification(
    llm: HF_Llama38b_LLM,
    embeddings: HF_MultilingualE5_Embeddings,
    memory: Memory,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
):
    # Parte 1: Resumiendo la conversacion
    output = generate_chat_summary(llm, memory, llm_collector)
    chat_summary = str(output["new_summary"]).strip()

    # Parte 2: Encontrando y generando terminos tecnicos de la conversacion
    terms_examples = retrieve_terms_examples(chat_summary, embeddings)
    output = generate_technical_terms(llm, llm_collector, chat_summary, terms_examples)
    technical_terms = output["terms"]

    # Parte 3: Recuperando definiciones para los terminos desde la bd vectorial (retrieval) para crear el diccionario
    terms_dictionary, has_replacement_definitions, _ = (
        retrieve_semantic_term_definitions(embeddings, technical_terms)
    )
    
    # Parte 4: Con el diccionario y el resumen, generamos un requerimiento más preciso
    output = generate_request_from_chat_summary(
        llm, memory, llm_collector, terms_dictionary
    )
    user_request = output["response"]
    complemented_user_request = None
    if has_replacement_definitions:
        # Parte 5: Identificando posible multidefinicion y claridad del requerimiento
        output = generate_multi_definition_detector(
            llm, llm_collector, user_request, terms_dictionary
        )

        # Parte 6: Verificando si el request no es ambiguo despues de ver las definiciones y multi definiciones
        complemented_user_request = generate_multi_definition_question(
            llm, llm_collector, user_request, output["class"], output["analysis"]
        )

        # Parte 7: Reemplazando los terminos especiales por estandares
        output = generate_flavored_request(
            llm, llm_collector, complemented_user_request, terms_dictionary
        )
        flavored_request_for_semantic_search = output["modified_sentence"]
    else:
        flavored_request_for_semantic_search = user_request
        complemented_user_request=user_request
    
    collector.conversation_summary = chat_summary
    collector.terms_dictionary = terms_dictionary
    collector.technical_terms = technical_terms
    collector.modified_user_request = complemented_user_request   
    collector.flavored_request_for_semantic_search = (
        flavored_request_for_semantic_search
    )

    return collector


def complex_request_process_semantics(
    llm: Base_LLM,
    embeddings: Openai_Embeddings,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
):
    flavored_request = collector.flavored_request_for_semantic_search
    modified_request = collector.modified_user_request
    # Busqueda semantica (retriever)
    semantic_tables, semantic_columns, semantic_relations_descriptions = (
        retrieve_sql_semantic_information(flavored_request, embeddings)
    )

    semantic_info = generate_semantic_info(
        model=llm,
        llm_collector=llm_collector,
        user_request=modified_request,
        semantic_tables=list(semantic_tables),
        semantic_columns=semantic_columns,
        semantic_relations_descriptions=list(semantic_relations_descriptions),
    )

    collector.semantic_info = semantic_info
    return collector
