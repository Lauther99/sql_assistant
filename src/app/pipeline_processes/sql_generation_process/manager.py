from src.app.pipeline_processes.sql_generation_process.generation import generate_sql


from src.app.pipeline_processes.sql_generation_process.retrievers import (
    retrieve_sql_examples,
)
from src.components.models.llms.llms import Langchain_OpenAI_LLM

from src.components.collector.collector import AppDataCollector, LLMResponseCollector


def complex_request_sql_generation(
    llm: Langchain_OpenAI_LLM,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
):
    user_request = collector.modified_user_request
    semantic_info = collector.semantic_info

    sql_examples = retrieve_sql_examples(user_request)

    output = generate_sql(llm, llm_collector, user_request, semantic_info, sql_examples, collector.terms_dictionary)

    collector.sql_code = output["sql_query"]
    return collector
