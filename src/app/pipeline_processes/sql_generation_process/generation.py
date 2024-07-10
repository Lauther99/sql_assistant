from src.app.pipeline_processes.sql_generation_process.prompts import (
    get_generate_sql_prompt,
)

# from src.app.pipeline_processes.sql_generation_process.retrievers import (
#     retrieve_sql_examples,
# )
from src.app.rag.rag_utils import base_llm_generation
from src.components.models.models_interfaces import Base_LLM
from src.components.collector.collector import LLMResponseCollector


def generate_sql(
    model: Base_LLM,
    llm_collector: LLMResponseCollector,
    user_request: str,
    semantic_info: dict[str, any],
    sql_examples,
    terms_dictionary: dict[str, any],
) -> str:
    # sql_examples = retrieve_sql_examples(user_request)
    instruction, suffix = get_generate_sql_prompt(
        user_request, sql_examples, semantic_info, terms_dictionary
    )
    prompt = model.apply_model_template(instruction, suffix)

    res = base_llm_generation(model, llm_collector, prompt, "generate-sql")
    return res
