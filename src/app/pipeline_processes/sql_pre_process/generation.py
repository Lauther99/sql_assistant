from src.app.pipeline_processes.sql_pre_process.retrievers import retrieve_sql_semantic_information
from src.app.rag.rag_utils import base_llm_generation
from src.components.models.models_interfaces import Base_LLM, Base_Embeddings
from src.app.pipeline_processes.sql_pre_process.prompts import get_generate_semantic_tables_prompt
from src.utils.utils import string_2_array


def generate_semantic_info(
    model: Base_LLM, embeddings: Base_Embeddings, user_request
) -> dict[str, any]:
    semantic_tables, semantic_columns, semantic_relations_descriptions = (
        retrieve_sql_semantic_information(user_request, embeddings)
    )
    input = get_generate_semantic_tables_prompt(
        user_request, list(semantic_tables), list(semantic_relations_descriptions)
    )
    res = base_llm_generation(model, input, "semantic-tables")
    res["tables"] = string_2_array(str(res["tables"]))

    semantic_info = {
        key: semantic_columns[key] for key in res["tables"] if key in semantic_columns
    }

    return semantic_info