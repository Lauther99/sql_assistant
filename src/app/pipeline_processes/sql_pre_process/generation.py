from src.app.pipeline_processes.sql_pre_process.retrievers import (
    retrieve_sql_semantic_information,
    retrieve_semantic_term_definitions,
)
from src.app.rag.rag_utils import base_llm_generation
from src.components.models.models_interfaces import Base_LLM, Base_Embeddings
from src.app.pipeline_processes.sql_pre_process.prompts import (
    get_generate_semantic_tables_prompt,
    get_multi_definition_question_prompt,
    get_complement_request_prompt,
    get_technical_terms_prompt,
    get_multi_definition_detector_prompt,
    get_modified_request_prompt,
)
from src.utils.utils import string_2_array


def generate_semantic_info(
    model: Base_LLM,
    user_request: str,
    semantic_tables: list,
    semantic_columns: dict,
    semantic_relations_descriptions: list
) -> dict[str, any]:

    instruction, suffix = get_generate_semantic_tables_prompt(
        user_request, list(semantic_tables), list(semantic_relations_descriptions)
    )

    prompt = model.apply_model_template(instruction, suffix)
    res = base_llm_generation(model, prompt, "semantic-tables")
    res["tables"] = string_2_array(str(res["tables"]))

    semantic_info = {
        key: semantic_columns[key] for key in res["tables"] if key in semantic_columns
    }

    return semantic_info


def generate_multi_definition_question(
    model: Base_LLM, user_request: str, isclear, isclear_analysis
):
    isclear = str(isclear).lower() == "unclear"
    if isclear:
        instruction, suffix = get_multi_definition_question_prompt(
            user_request, isclear_analysis
        )
        prompt = model.apply_model_template(instruction, suffix)
        output = base_llm_generation(model, prompt, "multi-definition-question")
        user_response = input(output["question"])

        instruction, suffix = get_complement_request_prompt(
            user_request, output["question"], user_response
        )

        prompt = model.apply_model_template(instruction, suffix)
        output = base_llm_generation(model, prompt, "final-request")

        return str(output["modified_sentence"])

    return user_request


def generate_technical_terms(
    model: Base_LLM,
    user_request: str,
):
    instruction, suffix = get_technical_terms_prompt(user_request)
    prompt = model.apply_model_template(instruction, suffix)
    output = base_llm_generation(model, prompt, "technical-terms")
    output["terms"] = string_2_array(output["terms"])

    return output


def generate_multi_definition_detector(
    model: Base_LLM, user_request: str, terms_dictionary
):
    instruction, suffix = get_multi_definition_detector_prompt(
        user_request, terms_dictionary
    )
    prompt = model.apply_model_template(instruction, suffix)
    output = base_llm_generation(model, prompt, "has-multi-definition")
    return output


def generate_flavored_request(model: Base_LLM, user_request: str, terms_dictionary):
    instruction, suffix = get_modified_request_prompt(user_request, terms_dictionary)
    prompt = model.apply_model_template(instruction, suffix)
    output = base_llm_generation(model, prompt, "modified-request")

    return output
