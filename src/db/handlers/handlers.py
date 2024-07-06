import sys
sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3")
from chromadb.api.models.Collection import Collection
from typing import Optional
from src.utils.utils import clean_sentence
from src.utils.reader_utils import read_database_semantics
import inspect


def query_by_texts(
    collection: Collection,
    texts: list[str],
    n: Optional[int] = 5,
    score_threshold: Optional[float] = 0.2,
    metadata_filters: Optional[dict[str, any]] = {},
):
    cleaned_texts = [clean_sentence(text) for text in texts]
    args, _, _, values = inspect.getargvalues(inspect.currentframe())

    if "metadata_filters" in args and values["metadata_filters"] is not None:
        results = collection.query(
            query_texts=cleaned_texts,
            n_results=n,
            include=["distances", "metadatas"],
            where=metadata_filters,
        )
    else:
        # El parámetro metadata_filters no ha sido proporcionado
        results = collection.query(
            query_texts=cleaned_texts,
            n_results=n,
            include=["distances", "metadatas"],
        )
    
    data = set()

    for distances, metadatas in zip(results["distances"], results["metadatas"]):
        for distance, metadata in zip(distances, metadatas):
            if distance <= score_threshold:
                metadata_tuple = tuple(metadata.items())
                data.add(metadata_tuple)

    return tuple(data)


def query_by_vector_embedding(
    collection: Collection,
    vector_embedding: list[float],
    n: Optional[int] = 5,
    score_threshold: Optional[float] = 0.8,
    metadata_filters: Optional[dict[str, any]] = {},
):
    """Querying a Chroma db by vector embedding"""
    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    if "metadata_filters" in args and values["metadata_filters"] is not None:
        results = collection.query(
            query_embeddings=[vector_embedding],
            n_results=n,
            include=["distances", "metadatas"],
            where=metadata_filters,
        )
    else:
        # El parámetro metadata_filters no ha sido proporcionado
        results = collection.query(
            query_embeddings=[vector_embedding],
            n_results=n,
            include=["distances", "metadatas"],
        )

    data = set()
    for distances, metadatas in zip(results["distances"], results["metadatas"]):
        for distance, metadata in zip(distances, metadatas):
            if 1 - distance >= score_threshold:
                metadata["distance"] = 1 - distance
                metadata_tuple = tuple(metadata.items())
                data.add(metadata_tuple)

    data_list = list(data)
    data_list.sort(key=lambda x: dict(x)["distance"], reverse=True)
    return tuple(data_list)


def process_searched_relations(
    items: tuple,
    tables_related: dict[str, any] = {},
    current_relations_descriptions: list[str] = [],
    index: int = 0,
) -> dict[str, any]:
    table_1_name = None
    table_2_name = None
    table_1_key = None
    table_2_key = None
    mid_table_name = None
    mid_table_key_1 = None
    mid_table_key_2 = None
    relation_description = None
    key_table_1_description = None
    key_table_2_description = None
    key_mid_table_1_description = None
    key_mid_table_2_description = None

    current_item = items[index]
    for item in current_item:
        if item[0] == "key_mid_table_1":
            mid_table_key_1 = None if item[1] == "" else item[1]
        elif item[0] == "key_mid_table_2":
            mid_table_key_2 = None if item[1] == "" else item[1]
        elif item[0] == "key_table_1":
            table_1_key = item[1]
        elif item[0] == "key_table_2":
            table_2_key = item[1]
        elif item[0] == "mid_table":
            mid_table_name = None if item[1] == "" else item[1]
        elif item[0] == "table_1":
            table_1_name = item[1]
        elif item[0] == "table_2":
            table_2_name = item[1]
        elif item[0] == "relation_description":
            relation_description = str(item[1]).strip()
        elif item[0] == "key_table_1_description":
            key_table_1_description = str(item[1]).strip()
        elif item[0] == "key_table_2_description":
            key_table_2_description = str(item[1]).strip()
        elif item[0] == "key_mid_table_1_description":
            key_mid_table_1_description = str(item[1]).strip()
        elif item[0] == "key_mid_table_2_description":
            key_mid_table_2_description = str(item[1]).strip()

    current_relations_descriptions.append(relation_description)

    tables_related.setdefault(table_1_name, set())
    tables_related.setdefault(table_2_name, set())
    if (
        mid_table_key_1 is None
        or mid_table_key_1 == ""
        and mid_table_key_2 is None
        or mid_table_key_2 == ""
        and mid_table_name is None
        or mid_table_name == ""
    ):
        tables_related[table_1_name].add(
            (
                table_1_key,
                "INT",
                key_table_1_description,
                999 if table_1_key != "Id" else 1,
                (table_1_name, table_2_name),
            )
        )
        tables_related[table_2_name].add(
            (
                table_2_key,
                "INT",
                key_table_2_description,
                999 if table_2_key != "Id" else 1,
                (table_1_name, table_2_name),
            )
        )

    if (
        mid_table_key_1 is not None
        and mid_table_key_2 is not None
        and mid_table_name is not None
    ):
        tables_related.setdefault(mid_table_name, set())
        tables_related[mid_table_name].add(
            (
                mid_table_key_1,
                "INT",
                key_mid_table_1_description,
                999 if mid_table_key_1 != "Id" else 1,
                (table_1_name, mid_table_name)
            )
        )
        tables_related[mid_table_name].add(
            (
                mid_table_key_2,
                "INT",
                key_mid_table_2_description,
                999 if mid_table_key_2 != "Id" else 1,
                (table_2_name, mid_table_name)
            )
        )
    current_relations_descriptions = set(current_relations_descriptions)
    if index < len(items) - 1:
        return process_searched_relations(
            items, tables_related, list(current_relations_descriptions), index + 1
        )
    else:
        return {
            "tables_related": tables_related,
            "table_relations_descriptions": list(current_relations_descriptions),
        }


def add_base_columns(
    tables: list[str], current_columns: dict[str, any]
) -> dict[str, any]:
    base_columns = read_database_semantics("basic_columns")
    filtered_columns = base_columns[base_columns["meta_table"].isin(tables)]
    for _, item in filtered_columns.iterrows():
        if item["meta_table"] not in current_columns.keys():
            current_columns[item["meta_table"]] = set()
        current_columns[item["meta_table"]].add(
            (
                item["meta_column_name"],
                item["meta_column_type"],
                str(item["meta_column_comment"]).strip(),
                item["meta_priority"],
                (item["meta_table"])
            )
        )

    return current_columns


def process_searched_columns(
    columns: tuple, current_columns: dict[str, any] = {}, index: int = 0
) -> dict[str, any]:
    table_name, column_name, column_type, column_description, column_priority = (
        None,
        None,
        None,
        None,
        None,
    )
    current_item = columns[index]
    for item in current_item:
        if item[0] == "meta_table":
            table_name = item[1]
        elif item[0] == "meta_column_name":
            column_name = item[1]
        elif item[0] == "meta_column_type":
            column_type = item[1]
        elif item[0] == "meta_column_comment":
            column_description = str(item[1]).strip()
        elif item[0] == "meta_priority":
            column_priority = item[1]

    current_columns.setdefault(table_name, set())
    current_columns[table_name].add(
        (column_name, column_type, column_description, column_priority, table_name)
    )
    if index < len(columns) - 1:
        return process_searched_columns(columns, current_columns, index + 1)
    else:
        return current_columns
