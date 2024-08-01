from typing import Any, Hashable
from src.components.collector.collector import LLMResponseCollector
from src.app.pipeline_processes.sql_post_process.prompts import (
    get_sql_pre_query_prompt,
    get_sql_classifier_prompt,
    get_sql_summary_response_prompt,
)
from src.app.rag.rag_utils import base_llm_generation
from src.components.models.models_interfaces import Base_LLM


def generate_sql_classification(
    model: Base_LLM,
    llm_collector: LLMResponseCollector,
    sql_code: str,
    semantic_info: dict[str, any],
):
    instruction, suffix = get_sql_classifier_prompt(sql_code, semantic_info)
    prompt = model.apply_model_template(instruction, suffix)
    res = base_llm_generation(model, llm_collector, prompt, "classify-sql")
    return res


def generate_sql_pre_query(
    model: Base_LLM,
    llm_collector: LLMResponseCollector,
    sql_code: str,
    assistant_analysis: str,
    assistant_suggestion: str,
):
    instruction, suffix = get_sql_pre_query_prompt(
        sql_code, assistant_analysis, assistant_suggestion
    )
    prompt = model.apply_model_template(instruction, suffix)
    res = base_llm_generation(model, llm_collector, prompt, "pre-query-sql")

    return res


def generate_sql_summary_response(
    model: Base_LLM,
    llm_collector: LLMResponseCollector,
    sql_dataframe: list[dict[Hashable, Any]],
    user_request: str,
    sql_code: str,
):
    instruction, suffix = get_sql_summary_response_prompt(
        sql_dataframe=sql_dataframe, user_request=user_request, sql_code=sql_code
    )
    prompt = model.apply_model_template(instruction, suffix)
    res = base_llm_generation(model, llm_collector, prompt, "summary-response-sql")

    return res
