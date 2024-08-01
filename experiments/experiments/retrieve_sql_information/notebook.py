import sys
sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3")

from src.components.models.embeddings.embeddings import HF_MultilingualE5_Embeddings, Openai_Embeddings
from src.components.models.llms.llms import HF_Llama38b_LLM, Openai_LLM

hf_llm = HF_Llama38b_LLM()
openai_llm = Openai_LLM()

# Modelos Embeddings
mle5_embeddings = HF_MultilingualE5_Embeddings()
openai_embeddings = Openai_Embeddings()

#Iniciando los modelos
openai_llm.init_model()
hf_llm.init_model()
mle5_embeddings.init_model()
openai_embeddings.init_model()


from src.db.chroma_db.handlers.handlers import add_base_columns, process_searched_relations, process_searched_columns, query_by_vector_embedding
from experiments.experiments_settings.settings import Experiments_Settings
from src.components.models.models_interfaces import Base_Embeddings
from src.utils.utils import clean_sentence



def retrieve_sql_semantic_information(
    user_request: str,
    embeddings: Base_Embeddings,
    nodes_search_properties: dict[str, any] = {"n": 5, "score_threshold": 0.7},
    relations_search_properties: dict[str, any] = {"n": 3, "score_threshold": 0.8},
    columns_search_properties: dict[str, any] = {"n": 4, "score_threshold": 0.7},
):
    """
    """
    nodes_collection = Experiments_Settings.Chroma.get_experiments_semantic_tables_collection()
    relations_collection = Experiments_Settings.Chroma.get_experiments_semantic_relations_collection()
    columns_collection = Experiments_Settings.Chroma.get_experiments_semantic_columns()

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
    tables_related_info = {}
    if len(results_relations_collection) > 0:
        tables_related_info = process_searched_relations(items=results_relations_collection, tables_related={})
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
        columns_1 = (
            list(tables_related_info["tables_related"].get(table, []))
            if tables_related_info
            else []
        )
        columns_2 = list(columns.get(table, []))
        total_columns = columns_1 + columns_2
        resultado_columnas[table] = set(total_columns)

        aux_list = list(resultado_columnas[table])
        ordered_list = sorted(aux_list, key=lambda x: x[3])
        resultado_columnas[table] = ordered_list
    return (
        tables,
        resultado_columnas,
        (
            tables_related_info["table_relations_descriptions"]
            if tables_related_info
            else []
        ),
    )


print("************************************ flow computers ************************************")
keyword = "flow computers"
r = retrieve_sql_semantic_information(keyword, openai_embeddings)
print(r[0])

print("\n\n************************************ measurement system ************************************")
keyword = "measurement system"
r = retrieve_sql_semantic_information(keyword, openai_embeddings)
print(r[0])

print("\n\n************************************ platform ************************************")
keyword = "platform"
r = retrieve_sql_semantic_information(keyword, openai_embeddings)
print(r[0])

