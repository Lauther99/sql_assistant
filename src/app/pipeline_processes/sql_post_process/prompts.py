import pandas as pd

create_table_template: str = (
    """CREATE TABLE IF NOT EXISTS dbo_v2.{table_name}(\n{list_columns_plus_type_plus_descriptions});"""
)

sql_classifier_template: str = """Your task is to classify the next SQL query into complete or incomplete.
Instructions:
Is complete when sql_query has all necessary to be a correct SQL query.
Is incomplete when you find placeholders in the query that needs to be replaced by user.
End of instructions

Database schema:
{tables}
End of Database schema:"""

sql_classifier_suffix = """Use the following key format to respond:
analysis: Brief analysis.
class: complete or incomplete.
suggestion: Brief recommendation for the user about the missing information to have better results.

Begin!
sql_query: '''{sql_query}''' """

generate_sql_pre_query_template: str = """The next is an incomplete SQL QUERY:
incomplete_sql_query: '''{incomplete_sql_query}'''
analysis: {analysis}
suggestion: {suggestion}

Your task is to generate a pre query where user can find this missing attributes."""

generate_sql_pre_query_suffix = """Use the following key format to respond:
analysis: Brief analysis.
sql_pre_query: SQL code.
tables: Correct comma separated list of used tables
intention: Very Brief intention of this sql_pre_query. Always respond the question "What is user requesting with sql_pre_query?". Must starts with: "The user is requesting ...".

Begin!"""

generate_summary_no_intents_template: str = """The following is a pandas DataFrame with the results of a database query:
{dataframe}

Your task is to generate a briefly summarize for data based on the user request that was asked. 
user_request: {user_request}

Do not respond with any additional explanation beyond the summary.
Do not include the pandas DataFrame in your response.
If Dataframe is empty, respond that answer is not available.
Ask user to select an item from the list if it is necessary."""

generate_summary_no_intents_suffix = """Note: You can make brief suggestion for user query to mention dates or names to retrieve better information from database.

Use the following key format to respond:
user_request: The asked user request.
response: Your briefly summarize.

Begin!:"""

generate_summary_with_intents_template: str = """The following is a user request:
user_request: {user_request}
But for before continue is necessary that user choose one option of the results in the pandas DataFrame:
{dataframe}

Your task is to cordially ask the user to choose something from the dataframe before continuing his request.
Do not include the specific options from dataframe in your response.
If Dataframe is empty, respond that his request can not be answered for the moment.
Do not include the pandas DataFrame in your response.
If you have to refer to something in the dataframe refer it as in 'the list'."""

generate_summary_with_intents_suffix = """Use the following key format to respond:
user_request: The asked user request.
response: Your briefly question.

Begin!"""


def get_sql_classifier_prompt(sql_query: str, semantic_info: dict[str, any]):
    ddls = list()

    for table in semantic_info:
        list_columns_plus_type_plus_descriptions = set()
        for item in semantic_info[table]:
            elementos = [item[4]] if isinstance(item[4], str) else list(item[4])
            if all(elemento in semantic_info for elemento in elementos):
                list_columns_plus_type_plus_descriptions.add(
                    (f"\t{item[0]} {item[1]} -- Description: {item[2]},\n", item[3])
                )
        # Crear una lista ordenada de las columnas basadas en la prioridad
        ordered_list = sorted(
            list_columns_plus_type_plus_descriptions, key=lambda x: x[1]
        )
        list_columns_plus_type_plus_descriptions = "".join(
            tupla[0] for tupla in ordered_list
        )
        t = create_table_template.format(
            table_name=table,
            list_columns_plus_type_plus_descriptions=list_columns_plus_type_plus_descriptions,
        )
        ddls.append(t)

    txt_tables = "\n\n".join(ddls)

    prompt = sql_classifier_template.format(tables=txt_tables)
    suffix = sql_classifier_suffix.format(sql_query=sql_query)
    return prompt, suffix


def get_sql_pre_query_prompt(incomplete_sql_query: str, analysis: str, suggestion: str):
    prompt = generate_sql_pre_query_template.format(
        incomplete_sql_query=incomplete_sql_query,
        analysis=analysis,
        suggestion=suggestion,
    )
    return prompt, generate_sql_pre_query_suffix


def get_sql_summary_response_prompt(sql_dataframe: pd.DataFrame, user_request: str, is_pre_query: bool):
    if not is_pre_query:
        prompt = generate_summary_no_intents_template
        prompt = prompt.format(
            dataframe=sql_dataframe.head(10).to_markdown(),
            user_request=user_request,
        )
        suffix = generate_summary_no_intents_suffix
    else:
        prompt = generate_summary_with_intents_template
        prompt = prompt.format(
            dataframe=sql_dataframe.head(10).to_markdown(),
            user_request=user_request,
        )
        suffix = generate_summary_with_intents_suffix
        
    return prompt, suffix
