from typing import Any, Hashable
from src.app.pipeline_processes.sql_post_process.generation import (
    generate_sql_classification,
    generate_sql_pre_query,
    generate_sql_summary_response,
)
from src.components.models.models_interfaces import Base_LLM
import pandas as pd
from src.utils.utils import clean_symbols
from src.components.collector.collector import AppDataCollector, LLMResponseCollector


def complex_request_sql_verification(
    llm: Base_LLM, collector: AppDataCollector, llm_collector: LLMResponseCollector
):
    sql_code = collector.sql_code
    semantic_info = collector.semantic_info

    output = generate_sql_classification(llm, llm_collector, sql_code, semantic_info)

    collector.assistant_sql_code_class = clean_symbols(str(output["class"]).strip())
    collector.assistant_sql_code_analysis = str(output["analysis"]).strip()
    collector.assistant_sql_code_suggestion = str(output["suggestion"]).strip()

    return collector


def complex_request_pre_query_generation(
    llm: Base_LLM, collector: AppDataCollector, llm_collector: LLMResponseCollector
):

    sql_code = collector.sql_code
    assistant_analysis = collector.assistant_sql_code_analysis
    assistant_suggestion = collector.assistant_sql_code_suggestion

    output = generate_sql_pre_query(
        llm, llm_collector, sql_code, assistant_analysis, assistant_suggestion
    )

    collector.sql_pre_query = output["sql_pre_query"]

    return collector


def complex_request_sql_summary_response(
    llm: Base_LLM,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
    dataframe: list[dict[Hashable, Any]],
):
    user_request = collector.modified_user_request
    pre_query = collector.sql_pre_query
    is_prequery = pre_query is not None and pre_query != ""

    output = generate_sql_summary_response(llm, llm_collector, dataframe, user_request, is_prequery)

    collector.ai_pre_response = output["response"]
    collector.dataframe_response = dataframe

    return collector
