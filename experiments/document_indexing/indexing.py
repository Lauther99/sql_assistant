import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3")
from experiments.utils.reader_utils import (
    read_database_experiments,
)
from src.utils.utils import clean_technical_term
from experiments.experiments_settings.settings import Experiments_Settings
from src.components.models.embeddings.embeddings import (
    Langchain_OpenAI_Embeddings,
    Openai_Embeddings,
    HF_MultilingualE5_Embeddings,
)
from langchain_community.vectorstores.chroma import (
    Chroma as Langchain_Chroma_Collection,
)
from experiments.experiments_settings.env_config import Config
from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    HuggingFaceEmbeddingFunction,
)
import chromadb

import pandas as pd


class ExperimentsIndexer:
    def __init__(self) -> None:
        # Iniciando modelos
        print("Iniciando modelos")
        self.init_models()

        # Configuracion cliente chromadb
        self.chroma_dev_db_path = Experiments_Settings.Chroma.get_db_path()
        self.chroma_experiments_db_path = Experiments_Settings.Chroma.get_db_path()

        self.chromadb_client = chromadb.PersistentClient(
            path=self.chroma_experiments_db_path
        )

        # Configuracion embeddings functions para las colecciones de chroma
        self.openai_embedding_function = OpenAIEmbeddingFunction(
            api_key=Experiments_Settings.Openai.get_api_key(),
            model_name=Experiments_Settings.Openai.get_embeddings_model(),
        )
        
        self.huggingface_embedding_function = HuggingFaceEmbeddingFunction(
            api_key=Experiments_Settings.Hugging_face.get_api_key(),
            model_name=Experiments_Settings.Hugging_face.get_embeddings_model_mle5(),
        )

        # Configuracion nombre de colecciones
        self.collection_names = {
            "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS": Config.get_experimentsdb_config()[
                "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"
            ],
            "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS": Config.get_experimentsdb_config()[
                "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"
            ],
        }

    def init_models(self):
        self.openai_native_model = Langchain_OpenAI_Embeddings()
        self.langchain_model = Openai_Embeddings()
        self.hf_model = HF_MultilingualE5_Embeddings()

        self.openai_native_model.init_model()
        self.langchain_model.init_model()
        self.hf_model.init_model()

    def delete_specific_collection(self) -> None:
        collection_names = {
            self.collection_names[
                "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"
            ]: lambda: self._delete_collection(
                self.collection_names["EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"]
            ),
            self.collection_names[
                "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"
            ]: lambda: self._delete_collection(
                self.collection_names["EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"]
            ),
        }
        print("Selecciona una colección:")
        for i, collection_name in enumerate(collection_names):
            print(f"{i+1}. {collection_name}")

        opcion = int(input("Ingresa el número de la colección que deseas eliminar: "))
        selected_collection = list(collection_names.keys())[opcion - 1]
        collection_names[selected_collection]()

    def _delete_collection(self, collection_name) -> None:
        print("Eliminando la coleccion ...")
        self.chromadb_client.delete_collection(name=collection_name)
        print("Eliminaste la coleccion: ", collection_name)

    # Training function
    def index_with_vectors(
        self,
        collection_name: str,
        ids,
        embedding_function,
        embeddings: list[list[float]],
        metadatas: list[dict] = [{"source": ""}],
    ):
        collection = self.chromadb_client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_function
        )

        collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

    def index_with_documents(
        self,
        collection_name: str,
        ids,
        embedding_function,
        documents: list[str],
        metadatas: list[dict] = [{"source": ""}],
    ):
        collection = self.chromadb_client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_function
        )

        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    # Training collections

    def train_openai_experiments_collection(self):
        # Leyendo el diccionario del excel
        excel_data = read_database_experiments(
            sheet_name="bussiness_semantics",
            cols=[
                "semantic_term_description",
                "meta_term",
                "meta_table_name",
                "meta_terms_definitions",
                "meta_terms_replacements",
                "meta_sql_advices",
            ],
        )
        vectors = []
        metadatas = []
        ids = []
        for index, row in excel_data.iterrows():
            processed_word = clean_technical_term(row["semantic_term_description"])
            new_embedding: list[float] = self.openai_native_model.get_embeddings(processed_word)
            # new_embedding: list[float] = get_embeddings(
            #     processed_word,
            #     self.openai_model,
            #     self.settings.openai.embeddings_model,
            # )
            # new_embedding = get_hf_embeddings(processed_word)
            vectors.append(new_embedding)
            metadatas.append(
                {
                    "meta_term": row["meta_term"],
                    "meta_table_name": row["meta_table_name"],
                    "meta_terms_definitions": (
                        " "
                        if pd.isna(row["meta_terms_definitions"])
                        else row["meta_terms_definitions"]
                    ),
                    "meta_terms_replacements": (
                        " "
                        if pd.isna(row["meta_terms_replacements"])
                        else row["meta_terms_replacements"]
                    ),
                }
            )
            ids.append(f"id_experiments_col_{index}")
        self.index_with_vectors(
            collection_name=self.collection_names[
                "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"
            ],
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas,
            embedding_function=self.openai_embedding_function
        )
        print("Colección openai_experiments entrenada!")

    def train_llama_experiments_collection(self):
        # Leyendo el diccionario del excel
        excel_data = read_database_experiments(
            sheet_name="bussiness_semantics",
            cols=[
                "semantic_term_description",
                "meta_term",
                "meta_table_name",
                "meta_terms_definitions",
                "meta_terms_replacements",
                "meta_sql_advices",
            ],
        )
        documents = []
        metadatas = []
        ids = []
        for index, row in excel_data.iterrows():
            processed_word = clean_technical_term(row["semantic_term_description"])
            documents.append(processed_word)
            metadatas.append(
                {
                    "meta_term": row["meta_term"],
                    "meta_table_name": row["meta_table_name"],
                    "meta_terms_definitions": (
                        " "
                        if pd.isna(row["meta_terms_definitions"])
                        else row["meta_terms_definitions"]
                    ),
                    "meta_terms_replacements": (
                        " "
                        if pd.isna(row["meta_terms_replacements"])
                        else row["meta_terms_replacements"]
                    ),
                }
            )
            ids.append(f"id_experiments_llama_col_{index}")
        self.index_with_documents(
            collection_name=self.collection_names[
                "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"
            ],
            ids=ids,
            embedding_function=self.huggingface_embedding_function,
            documents=documents,
            metadatas=metadatas,
        )
        print("Colección llama_experiments entrenada!")
    

    def train_specific_collection(self):
        collection_names = {
            self.collection_names[
                "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"
            ]: lambda: self.train_openai_experiments_collection(),
            self.collection_names[
                "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"
            ]: lambda: self.train_llama_experiments_collection(),
        }
        print("Selecciona una colección:")
        for i, collection_name in enumerate(collection_names):
            print(f"{i+1}. {collection_name}")

        opcion = int(input("Ingresa el número de la colección que deseas entrenar: "))
        selected_collection = list(collection_names.keys())[opcion - 1]
        collection_names[selected_collection]()

    # Extra
    def counting_vectors(self):
        for _, collection_value in self.collection_names.items():
            db = Langchain_Chroma_Collection(
                embedding_function=self.openai_embedding_function,
                persist_directory=self.chroma_dev_db_path,
                collection_name=collection_value,
            )
            count = db._collection.count()
            res = f"{collection_value}: {count} vectores"
            print(res)
