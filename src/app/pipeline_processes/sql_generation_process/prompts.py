from src.utils.reader_utils import read_database_semantics, read_tables_data
from collections import defaultdict

create_table_template = """CREATE TABLE IF NOT EXISTS dbo_v2.{table_name}(\n{list_columns_plus_type_plus_descriptions});"""

generate_sql_prefix_template: str = (
    """You are a SQL SERVER 2014 expert. Given an input request, create a syntactically correct SQL SERVER 2014 query to run. Remember NOT include backticks ```sql ``` before and after the created query. Unless otherwise specified."""
)

generate_sql_suffix: str = """Follow these Instructions for creating syntactically correct SQL SERVER 2014 query:
    - Be sure not to query for all columns, select more relevants ones. Avoid using SELECT * .
    - Be sure not to query for columns that do not exist in the tables and use alias only where required.
    - Likewise, when asked about the average (AVG function) or ratio, ensure the appropriate aggregation function is used.
    - Pay close attention to the filtering criteria mentioned in the question and incorporate them using the WHERE clause in your SQL query.
    - If the question involves multiple conditions, use logical operators such as AND, OR to combine them effectively.
    - When dealing with date or DATE columns, use appropriate date functions (e.g., DATEPART, GETDATE) for extracting specific parts of the date or performing date arithmetic.
    - If the question involves grouping of data (e.g., finding totals or averages for different categories), use the GROUP BY clause along with appropriate aggregate functions.
    - Consider using aliases for tables and columns to prevent ambiguities.
    
ANSWER FORMAT:
Use the following key format to respond:
sql_query: A correct SQL query.
suggestion: Brief recommendation for the user about the missing attributes or how to improve the natural_language_question to have better results.
used_tables: [List of table names used in 'sql_query'].
table_schema: Always use 'dbo_v2' schema.
END OF ANSWER FORMAT


input_request: '''{user_request}'''
Let's start thinking..."""

generate_sql_prefix_template: str = """I need your help to create a correct SQL SERVER 2014 query. Follow carefully the next steps:

First, Take your time and read carefully the next input request:
input_request: '''{user_request}'''

Next, Take your time and look up this past sql examples. Use them only to guide your answer, you are NOT allowed to use WHERE information from this examples in your response:
{examples}
 
Next, Look up this DDL from Database:
{ddl}

{information}

Next, generation. Take your time and generate a syntactically correct SQL SERVER 2014 query according to the previous information.

Next, evaluation. Evaluate if all the columns in your generated sql query are mentioned in the previous DDL list, if they are not, erase the column from your response. Make sure to use SQL SERVER 2014 statements for selecting or ordering.
Use this recommendations for a better evaluation:
- Be sure not to query for all columns, select more relevants ones. Avoid using SELECT * .
- Be sure not to query for columns that do not exist in the tables and use alias only where required.
- Likewise, when asked about the average (AVG function) or ratio, ensure the appropriate aggregation function is used.
- Pay close attention to the filtering criteria mentioned in the question and incorporate them using the WHERE clause in your SQL query.
- If the question involves multiple conditions, use logical operators such as AND, OR to combine them effectively.
- When dealing with date or DATE columns, use appropriate date functions (e.g., DATEPART, GETDATE) for extracting specific parts of the date or performing date arithmetic.
- If the question involves grouping of data (e.g., finding totals or averages for different categories), use the GROUP BY clause along with appropriate aggregate functions.
- Consider using aliases for tables and columns to prevent ambiguities.

Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
Start Key format:
"sql_query" is the key and its content is: A correct SQL SERVER 2014 query in ONE LINE, Do not make line breaks in the query.
"suggestion" is the key and its content is: A Brief recommendation for the user about the missing attributes or how to improve the natural_language_question to have better results.
"used_tables" is the key and its content is: Comma separated list of table names used in 'sql_query'].
End of Key format

Begin!"""

generate_sql_suffix: str = """sql_query: """


def _add_examples_in_prompt(sql_examples: tuple):
    if sql_examples:
        examples = "".join(
            f"input_request: {item[1][1]}\nsql_query: {item[0][1]}\n----------------------------------\n"
            for item in sql_examples
        )
        return examples
    return ""


def _add_documentation_in_prompt(
    terms_dictionary: dict[str, any], semantic_info: dict[str, any]
):
    documentation_info = defaultdict(list)
    has_sql_instructions = False
    information: str = ""
    
    for item in terms_dictionary:
        for inner_item in item["definitions"]:
            table_name = inner_item["table_name"]
            sql_instructions = inner_item["sql_instructions"].strip()

            if table_name in semantic_info and sql_instructions:
                has_sql_instructions = True
                documentation_info[table_name].append(sql_instructions)

    if has_sql_instructions:
        texts = "\n".join(
            f"Table info for: {table_name}\n    - " + "\n    - ".join(sqls)
            for table_name, sqls in documentation_info.items()
        )

        information += f"\nNext, Look up this table information:\n{texts}\n"

    return information


def _add_ddl_in_prompt(semantic_info: dict[str, any]):
    ddls = list()
    ddl_text = ""
    
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

    tables_context = "\n".join(ddls)
    ddl_text += f"\n{tables_context}\n\n"
    
    rel = read_database_semantics(
        "relations", ["table_1", "table_2", "join_description"]
    )
    for _, item in rel.iterrows():
        table_1_plus_table_2 = [item["table_1"], item["table_2"]]
        if all(elemento in semantic_info for elemento in table_1_plus_table_2):
            ddl_text += f"""{item["join_description"]}\n"""

    return ddl_text


def get_generate_sql_prompt(
    user_request: str,
    sql_examples: tuple,
    semantic_info: dict[str, any],
    terms_dictionary: dict[str, any],
):
    # Le agregamos los resultados de los ejemplos
    examples = _add_examples_in_prompt(sql_examples)

    # Le agregamos los DDL
    ddl = _add_ddl_in_prompt(semantic_info)

    # Le agregamos los resultados de la documentacion
    information = _add_documentation_in_prompt(terms_dictionary, semantic_info)

    # Agregamos la instruccion y suffix
    instruction = generate_sql_prefix_template.format(user_request=user_request, examples=examples, ddl=ddl, information=information)
    suffix = generate_sql_suffix

    return instruction, suffix

# def _add_examples_in_prompt(prompt: str, sql_examples: tuple):
#     if sql_examples:
#         examples = "".join(
#             f"input_request: {item[1][1]}\nsql_query: {item[0][1]}\n\n"
#             for item in sql_examples
#         )
#         prompt += f"\nThe next are past SQL QUERY examples. Use them only to guide your answer, do not use add information from examples in your response.\nEXAMPLES:\n{examples}END OF EXAMPLES\n"

#     return prompt


# def _add_documentation_in_prompt(prompt: str, terms_dictionary: dict[str, any], semantic_info: dict[str, any]):
#     documentation_info = defaultdict(list)
#     has_sql_instructions = False

#     for item in terms_dictionary:
#         for inner_item in item["definitions"]:
#             table_name = inner_item["table_name"]
#             sql_instructions = inner_item["sql_instructions"].strip()

#             if table_name in semantic_info and sql_instructions:
#                 has_sql_instructions = True
#                 documentation_info[table_name].append(sql_instructions)

#     if has_sql_instructions:
#         texts = "\n".join(
#             f"Table info for: {table_name}\n    - " + "\n    - ".join(sqls)
#             for table_name, sqls in documentation_info.items()
#         )
        
#         prompt += f"\nHere is some relevant table information:\nTABLES INFO:\n{texts}\nEND OF TABLES INFO\n"

#     return prompt


# def _add_ddl_in_prompt(prompt: str, semantic_info: dict[str, any]):
#     ddls = list()

#     for table in semantic_info:
#         list_columns_plus_type_plus_descriptions = set()
#         for item in semantic_info[table]:
#             elementos = [item[4]] if isinstance(item[4], str) else list(item[4])
#             if all(elemento in semantic_info for elemento in elementos):
#                 list_columns_plus_type_plus_descriptions.add(
#                     (f"\t{item[0]} {item[1]} -- Description: {item[2]},\n", item[3])
#                 )
#         # Crear una lista ordenada de las columnas basadas en la prioridad
#         ordered_list = sorted(
#             list_columns_plus_type_plus_descriptions, key=lambda x: x[1]
#         )
#         list_columns_plus_type_plus_descriptions = "".join(
#             tupla[0] for tupla in ordered_list
#         )
#         t = create_table_template.format(
#             table_name=table,
#             list_columns_plus_type_plus_descriptions=list_columns_plus_type_plus_descriptions,
#         )
#         ddls.append(t)

#     tables_context = "\n".join(ddls)
#     prompt += f"\nUse only tables names and Column names mentioned in DDL list:\nDDL LIST:\n{tables_context}\n\n"
#     rel = read_database_semantics(
#         "relations", ["table_1", "table_2", "join_description"]
#     )
#     for _, item in rel.iterrows():
#         table_1_plus_table_2 = [item["table_1"], item["table_2"]]
#         if all(elemento in semantic_info for elemento in table_1_plus_table_2):
#             prompt += f"""{item["join_description"]}\n"""

#     prompt += "END OF DDL LIST\n\nPay close attention on which column is in which table. Do not use columns from tables not mentioned in DDL list.\n\n"

#     return prompt


# def get_generate_sql_prompt(
#     user_request: str,
#     sql_examples: tuple,
#     semantic_info: dict[str, any],
#     terms_dictionary: dict[str, any]
# ):
#     prompt = generate_sql_prefix_template

#     # Le agregamos los resultados de los ejemplos
#     prompt = _add_examples_in_prompt(prompt, sql_examples)

#     # Le agregamos los DDL
#     prompt = _add_ddl_in_prompt(prompt, semantic_info)

#     # Le agregamos los resultados de la documentacion
#     prompt = _add_documentation_in_prompt(prompt, terms_dictionary, semantic_info)
    
#     # Agregamos el suffix
#     suffix = generate_sql_suffix.format(user_request=user_request)
    
#     return prompt, suffix
