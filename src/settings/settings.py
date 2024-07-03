import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3")
from src.settings.env_config import Config
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.api.models.Collection import Collection
import pyodbc

import os


class OpenAISettings:
    @staticmethod
    def get_api_key() -> str:
        return Config.get_openai_config()["OPENAI_API_KEY"]

    @staticmethod
    def get_llm_model_name() -> str:
        return Config.get_openai_config()["OPENAI_LLM_MODEL"]

    @staticmethod
    def get_embeddings_model() -> str:
        return Config.get_openai_config()["OPENAI_EMBEDDINGS_MODEL"]


class ChromaDBSetup:
    @staticmethod
    def get_db_path():
        return Config.get_chromadb_config()["CHROMADB_DIRECTORY"]

    @staticmethod
    def get_classify_col() -> Collection:
        """Context collection was created with chromadb library"""

        api_key = Config.get_openai_config()["OPENAI_API_KEY"]
        embeddings_model_name = Config.get_openai_config()["OPENAI_EMBEDDINGS_MODEL"]
        collection_name = Config.get_chromadb_config()["CLASSIFIER_COLLECTION"]

        chromadb_directory = Config.get_chromadb_config()["CHROMADB_DIRECTORY"]
        path = os.path.abspath(chromadb_directory)

        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embeddings_model_name,
        )
        chroma_client = chromadb.PersistentClient(path=path)
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=embedding_function
        )
        return collection

    @staticmethod
    def get_context_col() -> Collection:
        """Context collection was created with chromadb library"""
        api_key = Config.get_openai_config()["OPENAI_API_KEY"]
        embeddings_model_name = Config.get_openai_config()["OPENAI_EMBEDDINGS_MODEL"]

        collection_name = Config.get_chromadb_config()["CONTEXT_COLLECTION"]

        chromadb_directory = Config.get_chromadb_config()["CHROMADB_DIRECTORY"]
        path = os.path.abspath(chromadb_directory)

        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embeddings_model_name,
        )
        chroma_client = chromadb.PersistentClient(path=path)
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=embedding_function
        )
        return collection

    @staticmethod
    def get_sql_examples_collection() -> Collection:
        """"""
        api_key = Config.get_openai_config()["OPENAI_API_KEY"]
        embeddings_model_name = Config.get_openai_config()["OPENAI_EMBEDDINGS_MODEL"]

        collection_name = Config.get_chromadb_config()["SQL_EXAMPLES_COLLECTION"]

        chromadb_directory = Config.get_chromadb_config()["CHROMADB_DIRECTORY"]
        path = os.path.abspath(chromadb_directory)

        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embeddings_model_name,
        )
        chroma_client = chromadb.PersistentClient(path=path)
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=embedding_function
        )
        return collection

    @staticmethod
    def get_table_definitions_collection() -> Collection:
        """"""
        api_key = Config.get_openai_config()["OPENAI_API_KEY"]
        embeddings_model_name = Config.get_openai_config()["OPENAI_EMBEDDINGS_MODEL"]

        collection_name = Config.get_chromadb_config()["TABLE_DEFINITIONS_COLLECTION"]

        chromadb_directory = Config.get_chromadb_config()["CHROMADB_DIRECTORY"]
        path = os.path.abspath(chromadb_directory)

        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embeddings_model_name,
        )

        chroma_client = chromadb.PersistentClient(path=path)
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=embedding_function
        )
        return collection

    @staticmethod
    def get_relations_definitions_collection() -> Collection:
        """"""
        api_key = Config.get_openai_config()["OPENAI_API_KEY"]
        embeddings_model_name = Config.get_openai_config()["OPENAI_EMBEDDINGS_MODEL"]
        collection_name = Config.get_chromadb_config()[
            "RELATIONS_DEFINITIONS_COLLECTION"
        ]

        chromadb_directory = Config.get_chromadb_config()["CHROMADB_DIRECTORY"]
        path = os.path.abspath(chromadb_directory)

        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embeddings_model_name,
        )
        chroma_client = chromadb.PersistentClient(path=path)
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=embedding_function
        )
        return collection

    @staticmethod
    def get_columns_definitions_collection() -> Collection:
        """"""

        api_key = Config.get_openai_config()["OPENAI_API_KEY"]
        embeddings_model_name = Config.get_openai_config()["OPENAI_EMBEDDINGS_MODEL"]

        collection_name = Config.get_chromadb_config()["COLUMNS_DEFINITIONS_COLLECTION"]

        chromadb_directory = Config.get_chromadb_config()["CHROMADB_DIRECTORY"]
        path = os.path.abspath(chromadb_directory)

        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embeddings_model_name,
        )
        chroma_client = chromadb.PersistentClient(path=path)
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=embedding_function
        )
        return collection


class ChromaExperimentsDBSetup:
    @staticmethod
    def get_db_path():
        return Config.get_chromadb_config()["CHROMADB_EXPERIMENTS_DIRECTORY"]

    @staticmethod
    def get_experiments_with_llama_collection() -> Collection:
        """"""

        api_key = Config.get_hf_config()["HF_KEY"]
        embeddings_model_name = Config.get_hf_config()[
            "HF_INFLOAT_MLE5_EMBEDDINGS_MODEL"
        ]

        collection_name = Config.get_experimentsdb_config()[
            "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"
        ]

        chromadb_directory = Config.get_experimentsdb_config()[
            "CHROMADB_EXPERIMENTS_DIRECTORY"
        ]
        path = os.path.abspath(chromadb_directory)

        embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=api_key,
            model_name=embeddings_model_name,
        )
        chroma_client = chromadb.PersistentClient(path=path)
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=embedding_function
        )
        return collection

    @staticmethod
    def get_experiments_with_openai_collection() -> Collection:
        """"""
        api_key = Config.get_openai_config()["OPENAI_API_KEY"]
        embeddings_model_name = Config.get_openai_config()["OPENAI_EMBEDDINGS_MODEL"]

        collection_name = Config.get_experimentsdb_config()[
            "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"
        ]

        chromadb_directory = Config.get_experimentsdb_config()[
            "CHROMADB_EXPERIMENTS_DIRECTORY"
        ]
        path = os.path.abspath(chromadb_directory)

        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embeddings_model_name,
        )

        chroma_client = chromadb.PersistentClient(path=path)

        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=embedding_function
        )
        return collection


class SQLSettings:
    @staticmethod
    def get_pyodbc_connection_string():
        db_user: str = Config.get_sqldatabase_config()["DB_USER"]
        db_pwd: str = Config.get_sqldatabase_config()["DB_PWD"]
        db_host: str = Config.get_sqldatabase_config()["DB_HOST"]
        db_name: str = Config.get_sqldatabase_config()["DB_NAME"]
        db_driver: str = Config.get_sqldatabase_config()["DB_DRIVER"]
        pyodbc_connection: pyodbc.Connection = pyodbc.connect(
            f"DRIVER={db_driver};SERVER={db_host};DATABASE={db_name};UID={db_user};PWD={db_pwd}"
        )
        return pyodbc_connection


class HFSettings:
    @staticmethod
    def get_api_key() -> str:
        return Config.get_hf_config()["HF_KEY"]

    @staticmethod
    def get_llm_model_llama3_8b() -> str:
        return Config.get_hf_config()["HF_META_LLAMA_LLAMA38B_MODEL"]

    @staticmethod
    def get_embeddings_model_mle5() -> str:
        return Config.get_hf_config()["HF_INFLOAT_MLE5_EMBEDDINGS_MODEL"]

class Settings:
    Openai: OpenAISettings
    Chroma: ChromaDBSetup
    Sql: SQLSettings
    Hugging_face: HFSettings

class Experiments_Settings:
    Openai: OpenAISettings
    Chroma: ChromaExperimentsDBSetup
    Sql: SQLSettings
    Hugging_face: HFSettings
