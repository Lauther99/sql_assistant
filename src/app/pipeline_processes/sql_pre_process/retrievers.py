from src.components.models.models_interfaces import Base_Embeddings
from src.db.handlers.handlers import (
    add_base_columns,
    process_searched_columns,
    process_searched_relations,
    query_by_vector_embedding,
)
from src.settings.settings import Settings
from src.utils.utils import clean_sentence, clean_technical_term
from chromadb.api.models.Collection import Collection


def retrieve_sql_semantic_information(
    user_request: str,
    embeddings: Base_Embeddings,
    nodes_search_properties: dict[str, any] = {"n": 5, "score_threshold": 0.6},
    relations_search_properties: dict[str, any] = {"n": 3, "score_threshold": 0.5},
    columns_search_properties: dict[str, any] = {"n": 4, "score_threshold": 0.6},
):
    """
    Realiza una búsqueda semántica utilizando un vector generado a partir de un `user_request`. Ajusta las propiedades de los nodos y relaciones para refinar la búsqueda según los parámetros dados o por defecto.
    ### Parámetros
    - **user_request** (`str`): La solicitud del usuario que se utiliza para generar el vector de búsqueda semántica.
    - **open_ai_model** (`OpenAI`): El modelo de OpenAI que se utiliza para procesar la solicitud del usuario y generar el vector de búsqueda.
    - **open_ai_model_name** (`str`): El nombre del modelo de OpenAI que se utilizará.
    - **nodes_collection** (`Collection`): La colección de nodos donde se realizará la búsqueda.
    - **relations_collection** (`Collection`): La colección de relaciones donde se realizará la búsqueda.
    - **nodes_properties** (`dict[str, any]`, opcional): Propiedades para seleccionar el vector mas cercano. Valores por defecto `{'n': 5, 'score_threshold': 0.6}`.
    - **relations_properties** (`dict[str, any]`, opcional):Propiedades para seleccionar el vector mas cercano. Valores por defecto `{'n': 3, 'score_threshold': 0.5}`.
    - `'n'` (`int`): Número de relaciones a considerar en los resultados.
    - `'score_threshold'` (`float`): Umbral de puntuación para filtrar las relaciones.
    ### Retorno
    - **tuple[list[str], list[str]]**:
    Una tupla que contiene dos listas:
    - `list[str]`: Lista de nombres de tablas que cumplen con los criterios de búsqueda semántica.
    - `list[str]`: Lista de relaciones que cumplen con los criterios de búsqueda semántica.
    ### Ejemplo de uso
    ```python
    tables, relations, columns = get_tables_with_semantic_search_by_vector(
        user_request="Encuentra las tablas relacionadas con ventas y clientes",
        open_ai_model=open_ai_model_instance,
        open_ai_model_name="text-davinci-003",
        nodes_collection=nodes_collection_instance,
        relations_collection=relations_collection_instance,
    )
    """
    nodes_collection = Settings.Chroma.get_table_definitions_collection()
    relations_collection = Settings.Chroma.get_relations_definitions_collection()
    columns_collection = Settings.Chroma.get_columns_definitions_collection()

    tables = set()
    vector = embeddings.get_embeddings(clean_sentence(user_request))

    # Buscamos las tablas como nodos
    results_table_collection = query_by_vector_embedding(
        collection=nodes_collection,
        vector_embedding=vector,
        n=nodes_search_properties["n"],
        score_threshold=nodes_search_properties["score_threshold"],
    )
    nodes = list({item[0][1] for item in results_table_collection})

    # Buscamos las tablas en las relaciones de las tablas anteriores
    # Busqueda en table_1
    results_relations_collection = query_by_vector_embedding(
        collection=relations_collection,
        vector_embedding=vector,
        n=relations_search_properties["n"],
        score_threshold=relations_search_properties["score_threshold"],
        metadata_filters={"table_1": {"$in": nodes}},
    )
    # Busqueda en table_2
    results_relations_collection += query_by_vector_embedding(
        collection=relations_collection,
        vector_embedding=vector,
        n=relations_search_properties["n"],
        score_threshold=relations_search_properties["score_threshold"],
        metadata_filters={"table_2": {"$in": nodes}},
    )

    # Actualizamos las tablas
    tables.update(nodes)
    tables_related_info = process_searched_relations(results_relations_collection)
    tables = set(tables_related_info["tables_related"].keys()).union(tables)

    # Agregamos las columnas a las tablas
    results_columns_collection = tuple()
    for table in tables:
        metadata_filters = {"meta_table": {"$eq": table}}
        results_columns_collection += query_by_vector_embedding(
            collection=columns_collection,
            vector_embedding=vector,
            n=columns_search_properties["n"],
            score_threshold=columns_search_properties["score_threshold"],
            metadata_filters=metadata_filters,
        )
    current_columns = {}
    if len(results_columns_collection) > 0:
        current_columns = process_searched_columns(results_columns_collection)
    columns = add_base_columns(tables, current_columns)

    resultado_columnas = {}
    for table in tables:
        columns_1 = list(tables_related_info["tables_related"].get(table, []))
        columns_2 = list(columns.get(table, []))
        total_columns = columns_1 + columns_2
        resultado_columnas[table] = set(total_columns)

        aux_list = list(resultado_columnas[table])
        ordered_list = sorted(aux_list, key=lambda x: x[3])
        resultado_columnas[table] = ordered_list

    return (
        tables,
        resultado_columnas,
        tables_related_info["table_relations_descriptions"],
    )


def retrieve_semantic_term_definitions(
    embeddings: Base_Embeddings, terms_collection: Collection, technical_terms_arr
):
    tables = [
        "teq_clasificacion",
        "teq_tipo_equipo",
        "equ_equipo",
        "med_tag",
        "pla_plataforma",
        "med_sistema_medicion",
        "fcs_computador_medidor",
        "fcs_firmware",
        "var_tipo_variable",
        "var_variable_datos",
        "flu_tipo_fluido",
        "fcs_computadores",
        "fcs_tipo_computador",
        "med_tipo_medicion",
    ]
    terms_arr = list()
    has_replacement_definitions = False
    sql_instructions = False

    for term in technical_terms_arr:
        original_term = term
        cleaned_term = clean_technical_term(clean_sentence(original_term))

        vector = embeddings.get_embeddings(cleaned_term)

        definitions = list()
        for table in tables:
            r = query_by_vector_embedding(
                collection=terms_collection,
                vector_embedding=vector,
                n=1,
                score_threshold=0.85,
                metadata_filters={"meta_table_name": {"$eq": table}},
            )
            for item in r:
                if item[2][1].strip():
                    definitions.append(
                        {
                            "standard_term": item[1][1],
                            "definition": item[2][1],
                            "replace_instruction": item[3][1],
                            # "distance": item[4][1],
                        }
                    )
                    has_replacement_definitions = (
                        True if item[3][1].strip() else has_replacement_definitions
                    )
                    # sql_instructions = True if item[4][1].strip() else sql_instructions

        if definitions:
            terms_arr.append(
                {
                    "original_term": original_term,
                    "cleaned_term": cleaned_term,
                    "definitions": definitions,
                }
            )

    return terms_arr, has_replacement_definitions, sql_instructions
