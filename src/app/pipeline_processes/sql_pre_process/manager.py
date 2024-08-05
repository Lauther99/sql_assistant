from src.components.models.models_interfaces import Base_LLM
from src.components.memory.memory import Memory
from src.app.pipeline_processes.sql_pre_process.retrievers import (
    retrieve_semantic_term_definitions,
    retrieve_sql_semantic_information,
    retrieve_sql_semantic_information_improved,
    retrieve_terms_examples,
)
from src.app.pipeline_processes.query_pre_process.manager import query_pre_process
from src.app.pipeline_processes.sql_pre_process.generation import (
    generate_flavored_request,
    generate_multi_definition_detector,
    generate_multi_definition_question,
    generate_enhanced_request,
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
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
):
    
    # Parte 1: Obtenemos el user request del collector
    # output = generate_enhanced_request(llm, llm_collector, collector)
    # user_request = output["response"]
    user_request = collector.modified_user_request
    
    # Parte 2: Encontrando y generando terminos tecnicos de la conversacion
    terms_examples = retrieve_terms_examples(user_request, embeddings)
    output = generate_technical_terms(llm, llm_collector, user_request, terms_examples)
    technical_terms = output["terms"]

    # Parte 3: Recuperando definiciones para los terminos (retrieval) para crear el diccionario
    terms_dictionary, _, _ = (
        retrieve_semantic_term_definitions(embeddings, technical_terms)
    )

    output = generate_flavored_request(llm, llm_collector, terms_dictionary)
    semantic_list_terms = output["response"] if output["response"] else technical_terms
    
    # if has_replacement_definitions:
        # Parte 5: Identificando posible multidefinicion y claridad del requerimiento
        # output = generate_multi_definition_detector(
        #     llm, llm_collector, user_request, terms_dictionary
        # )

        # Parte 6: Verificando si el request no es ambiguo despues de ver las definiciones y multi definiciones
        # modified_user_request = generate_multi_definition_question(
        #     llm, llm_collector, user_request, output["class"], output["analysis"]
        # )

        # Parte 7: Reemplazando los terminos especiales por estandares
        # output = generate_flavored_request(
        #     llm, llm_collector, modified_user_request, terms_dictionary
        # )
    #     output = generate_flavored_request(llm, llm_collector, terms_dictionary)
    #     semantic_list_terms = output["response"]
    # else:
    #     semantic_list_terms = technical_terms

    collector.terms_dictionary = terms_dictionary
    collector.technical_terms = technical_terms
    collector.semantic_list_terms = (
        semantic_list_terms
    )

    return collector


def complex_request_process_semantics(
    llm: Base_LLM,
    embeddings: Openai_Embeddings,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
):
    kewywords_arr = collector.semantic_list_terms
    modified_request = collector.modified_user_request
    
    # Busqueda semantica (retriever)
    semantic_tables, semantic_columns, semantic_relations_descriptions = (
        retrieve_sql_semantic_information_improved(kewywords_arr, embeddings)
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
