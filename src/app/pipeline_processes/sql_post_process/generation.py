from src.app.pipeline_processes.sql_post_process.prompts import (
    get_sql_pre_query_prompt,
    get_sql_classifier_prompt,
    get_sql_summary_response_prompt,
)
from src.app.rag.rag_utils import base_llm_generation
from src.components.models.models_interfaces import Base_LLM
import pandas as pd

def generate_sql_classification(
    model: Base_LLM,
    sql_code: str,
    semantic_info: dict[str, any],
):
    input = get_sql_classifier_prompt(sql_code, semantic_info)
    res = base_llm_generation(model, input, "classify-sql")

    return res


def generate_sql_pre_query(
    model: Base_LLM,
    sql_code: str,
    assistant_analysis: str,
    assistant_suggestion: str,
):
    input = get_sql_pre_query_prompt(sql_code, assistant_analysis, assistant_suggestion)
    res = base_llm_generation(model, input, "pre-query-sql")
    
    return res

def generate_sql_summary_response(
    model: Base_LLM,
    sql_dataframe: pd.DataFrame,
    user_request: str,
    is_pre_query: bool,
):
    input = get_sql_summary_response_prompt(sql_dataframe, user_request, is_pre_query)
    res = base_llm_generation(model, input, "summary-response-sql")
    
    return res
    
    