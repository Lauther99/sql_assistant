from src.app.pipeline_processes.sql_pre_process.retrievers import (
    retrieve_semantic_term_definitions,
    retrieve_sql_semantic_information,
)
from src.app.pipeline_processes.query_pre_process.manager import query_pre_process
from src.app.pipeline_processes.sql_pre_process.generation import (
    generate_flavored_request,
    generate_multi_definition_detector,
    generate_multi_definition_question,
    generate_semantic_info,
    generate_technical_terms,
)
from src.components.models.llms.llms import HF_Llama38b_LLM, Langchain_OpenAI_LLM
from src.settings.settings import Settings
from src.components.models.embeddings.embeddings import (
    HF_MultilingualE5_Embeddings,
    Openai_Embeddings,
)
from src.app.app import AppDataCollector


def complex_request_process_modification(
    llm: HF_Llama38b_LLM,
    embeddings: HF_MultilingualE5_Embeddings,
    collector: AppDataCollector,
):
    # Parte 1: Generando terminos tecnicos
    user_request = collector.user_request
    output = generate_technical_terms(llm, user_request)
    technical_terms = output["terms"]

    # Parte 2: Recuperando definiciones para los terminos desde la bd vectorial (retrieval)
    terms_collection = Settings.Chroma.get_terms_collection()
    terms_dictionary, has_replacement_definitions, _ = (
        retrieve_semantic_term_definitions(
            embeddings, terms_collection, technical_terms
        )
    )

    if has_replacement_definitions:
        # Parte 3: Identificando posible multidefinicion y claridad de la consulta
        output = generate_multi_definition_detector(llm, user_request, terms_dictionary)

        # Parte 4: Verificando si el request no es ambiguo despues de ver las definiciones y multi definiciones
        complemented_user_request = generate_multi_definition_question(
            llm, user_request, output["class"], output["analysis"]
        )

        # Parte 5: Reemplazando los terminos especiales por estandares
        flavored_request_for_semantic_search = generate_flavored_request(
            llm, complemented_user_request, terms_dictionary
        )
    else:
        flavored_request_for_semantic_search = user_request

    collector.terms_dictionary = terms_dictionary
    collector.technical_terms = technical_terms
    collector.modified_user_request = complemented_user_request
    collector.flavored_request_for_semantic_search = (
        flavored_request_for_semantic_search
    )

    return collector


def complex_request_process_semantics(
    llm: Langchain_OpenAI_LLM,
    embeddings: Openai_Embeddings,
    collector: AppDataCollector,
):
    modified_request = collector.flavored_request_for_semantic_search
    
    # Busqueda semantica (retriever)
    semantic_tables, semantic_columns, semantic_relations_descriptions = (
        retrieve_sql_semantic_information(modified_request, embeddings)
    )

    semantic_info = generate_semantic_info(
        llm,
        modified_request,
        list(semantic_tables),
        semantic_columns,
        list(semantic_relations_descriptions),
    )

    collector.semantic_info = semantic_info
    return collector
