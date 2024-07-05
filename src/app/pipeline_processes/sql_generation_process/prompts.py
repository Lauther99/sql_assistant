from src.utils.reader_utils import read_database_semantics, read_tables_data

generate_sql_prefix_template: str = """You are a SQL SERVER 2014 expert. Given an input request, create a syntactically correct SQL SERVER 2014 query to run. Remember NOT include backticks ```sql ``` before and after the created query. Unless otherwise specified."""

generate_sql_suffix_template: str = """Follow these Instructions for creating syntactically correct SQL query:
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

create_table_template = """CREATE TABLE IF NOT EXISTS dbo_v2.{table_name}(\n{list_columns_plus_type_plus_descriptions});"""


def add_examples_in_prompt(prompt: str, sql_examples: tuple):
    if sql_examples:
        examples = "".join(
            f"input_request: {item[1][1]}\nsql_query: {item[0][1]}\n\n"
            for item in sql_examples
        )
        prompt += f"\nThe next are past SQL QUERY examples. Use them only to guide your answer, do not use add information from examples in your response.\nEXAMPLES:\n{examples}END OF EXAMPLES\n"

    return prompt


def add_documentation_in_prompt(prompt: str, semantic_info: dict[str, any]):
    documentation = []
    _, doc_df = read_tables_data()
    grouped_doc = doc_df.groupby("table")["documentation"].apply(list).reset_index()
    # Preparar la documentación de las tablas, si existe
    for table in semantic_info:
        doc_data = grouped_doc[grouped_doc["table"] == table].to_dict("records")
        if doc_data:
            documentation.extend(doc_data)

    if documentation:
        examples = "\n".join(
            [
                f"Table info for: {doc['table']}\n    - "
                + "\n    - ".join(doc["documentation"])
                + "\n"
                for doc in documentation
            ]
        )
        prompt += f"\nHere is the relevant table info:\nTABLES INFO:\n{examples}\nEND OF TABLES INFO\n"

    return prompt


def add_ddl_in_prompt(prompt: str, semantic_info: dict[str, any]):
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

    tables_context = "\n".join(ddls)
    prompt += f"\nUse only tables names and Column names mentioned in context:\nCONTEXT:\n{tables_context}\n\n"
    rel = read_database_semantics(
        "relations", ["table_1", "table_2", "join_description"]
    )
    for _, item in rel.iterrows():
        table_1_plus_table_2 = [item["table_1"], item["table_2"]]
        if all(elemento in semantic_info for elemento in table_1_plus_table_2):
            prompt += f"""{item["join_description"]}\n"""

    prompt += "END OF CONTEXT\n\nPay close attention on which column is in which table. Do not use columns from tables not mentioned in context.\n\n"

    return prompt


def get_generate_sql_prompt(
    user_request: str,
    sql_examples: tuple,
    semantic_info: dict[str, any],
):
    prompt = generate_sql_prefix_template

    # Le agregamos los resultados de los ejemplos
    prompt = add_examples_in_prompt(prompt, sql_examples)

    # Le agregamos los resultados de la documentacion
    prompt = add_documentation_in_prompt(prompt, semantic_info)

    # Le agregamos los DDL
    prompt = add_ddl_in_prompt(prompt, semantic_info)

    # Agregamos el suffix
    prompt += generate_sql_suffix_template.format(user_request=user_request)
    
    return prompt
