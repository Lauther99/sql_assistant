from src.app.pipeline_processes.sql_generation_process.prompts import (
    get_generate_sql_prompt,
)
from src.app.pipeline_processes.sql_generation_process.retrievers import (
    retrieve_sql_examples,
)
from src.app.rag.rag_utils import base_llm_generation
from src.components.models.models_interfaces import Base_LLM


def generate_sql(
    model: Base_LLM,
    user_request: str,
    semantic_info: dict[str, any],
) -> str:
    sql_examples = retrieve_sql_examples(user_request)
    input = get_generate_sql_prompt(user_request, sql_examples, semantic_info)

    res = base_llm_generation(model, input, "generate-sql")
    
    return res
