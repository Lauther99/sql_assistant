from typing import Any, Hashable
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
End of Database schema"""

sql_classifier_suffix = """Note: A placeholder looks like: '<Name of placeholder>'

Use the following key format to respond:
analysis: Brief analysis.
class: complete or incomplete.
suggestion: Brief recommendation for the user about the missing information to have better results.

Begin!
sql_query: '''{sql_query}''' """

sql_classifier_template: str = """Your task is to classify the next SQL query into complete or incomplete.
Instructions:
Is complete when sql_query has all necessary to be a correct SQL query.
Is incomplete when you find placeholders in the query that needs to be replaced by user.
End of instructions

Database schema:
{tables}
End of Database schema.
Note: A placeholder looks like: '<Name of placeholder>'. It is not necessary to JOIN all the tables.


Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
Start Key format:
- "class" is the key and its content is: complete or incomplete.
- "analysis" is the key and its content is: Your brief analysis in one line, do not make break lines.
- "suggestion" is the key and its content is: A brief recommendation for the user about the missing information to have better results, do not make break lines.
- "suggested_sql" is the key and its content is: SQL SERVER 2014 code query sugested in one line, do not make break lines.
End of Key format

Begin!
sql_query: '''{sql_query}'''"""

sql_classifier_suffix = """class:"""

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

generate_summary_no_intents_template: str = """I need your help. Follow carefully the next steps:

First, look up the next user request:
<User request>{user_request}</User request>

Second, the next is a pandas dataframe that answers the previous request. Pay attention to this information:
<Dataframe>
{dataframe}
</Dataframe>

Third, generation. Generate a brief response to the user request based on the previous dataframe.

Fourth, evaluation. Evaluate if your response is answering the user request.

Note:
 - Do not include any explanations or apologies in your response.
 - Do not add your own conclusions or clarifications.
 - If dataframe is empty, say that the response is not available."""

generate_summary_no_intents_suffix = """
Use the following key format to respond:
response: Your briefly response in one line, do not make line breaks.

Begin!"""

generate_summary_with_intents_template: str = """The following is a user request:
<User request>{user_request}</User request>
But for before continue is necessary that user choose one option of the results in the pandas DataFrame:
<Dataframe>
{dataframe}
</Dataframe>

Your task is to cordially ask the user to choose something from the dataframe before continuing his request.
Do not include the specific options from dataframe in your response.
If Dataframe is empty, respond that his request can not be answered for the moment.
Do not include the pandas DataFrame in your response.
If you have to refer to something in the dataframe refer it as in 'the list'."""

generate_summary_with_intents_suffix = """Use the following key format to respond:
response: Your briefly question in one line, do not make line breaks.

Begin!"""

improved_summary_instruction: str = """According to this user request:
<Request>
{user_request}
</Request>

This is an sql code:
<Sql>
{sql_code}
</Sql>

And this answer from database:
<dataframe>
{dataframe}
</dataframe>

As an expert in SQL, your task is to understand the previous given information and explain briefly the response obtained for the user.

Note: 
User is not an expert, so he does not need a lot of technical sql details.
But you can explain understandable filters like dates or status.

Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
Start Key format:
"response" is the key and its content is: Brief analysis for normal user in one line, do not make line breaks.
End of Key format

Begin!"""

improved_summary_suffix = """response:"""


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

    prompt = sql_classifier_template.format(tables=txt_tables, sql_query=sql_query)
    suffix = sql_classifier_suffix
    return prompt, suffix


def get_sql_pre_query_prompt(incomplete_sql_query: str, analysis: str, suggestion: str):
    prompt = generate_sql_pre_query_template.format(
        incomplete_sql_query=incomplete_sql_query,
        analysis=analysis,
        suggestion=suggestion,
    )
    return prompt, generate_sql_pre_query_suffix


def get_sql_summary_response_prompt(
    sql_dataframe: list[dict[Hashable, Any]], user_request: str, sql_code: str
):
    prompt = improved_summary_instruction.format(
        dataframe=pd.DataFrame(sql_dataframe).head(10).to_markdown(),
        user_request=user_request,
        sql_code=sql_code,
    )
    suffix = improved_summary_suffix
    return prompt, suffix

    # if not is_pre_query:
    #     prompt = generate_summary_no_intents_template
    #     prompt = prompt.format(
    #         dataframe=pd.DataFrame(sql_dataframe).head(10).to_markdown(),
    #         user_request=user_request,
    #     )
    #     suffix = generate_summary_no_intents_suffix
    # else:
    #     prompt = generate_summary_with_intents_template
    #     prompt = prompt.format(
    #         dataframe=pd.DataFrame(sql_dataframe).head(10).to_markdown(),
    #         user_request=user_request,
    #     )
    #     suffix = generate_summary_with_intents_suffix

    # return prompt, suffix
