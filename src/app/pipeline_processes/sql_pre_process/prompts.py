from src.components.models.models_interfaces import Base_Embeddings
from src.utils.reader_utils import read_tables_descriptions


related_semantic_tables_template: str = """According to this tables descriptions:
{descriptions}
Use this relationships descriptions to find the most related tables to the user request:
{relations_descriptions_text}

Your task is to pick the most related tables according to the next user request:
user_request: {user_request}

Note: Yoy have to answer with database table name instead of the name of the table.

Use the following key format to respond:
tables: Comma separated list of selected tables.

Begin!"""


def get_generate_semantic_tables_prompt(
    user_request,
    semantic_tables,
    semantic_relations_descriptions,
):
    data = read_tables_descriptions()
    relevant_descriptions = [
        (d["table_name"], d["descriptions"], d["aka_name"])
        for d in data
        if d["table_name"] in semantic_tables
    ]
    descriptions = "\n".join(
        [
            f"{doc[2]} table: {doc[1]}.\nDatabase table name: {doc[0]}"
            for doc in relevant_descriptions
        ]
    )
    relations_descriptions_text = "\n".join(
        [f" - {doc}" for doc in semantic_relations_descriptions]
    )
    prompt = related_semantic_tables_template.format(
        descriptions=descriptions,
        relations_descriptions_text=relations_descriptions_text,
        user_request=user_request,
    )
    
    return prompt
