import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\open_ai_assistant_v2")
from src.utils.reader_utils import (
    read_sql_examples,
    read_classify_dictionary,
    read_database_semantics,
    read_database_experiments,
)
from src.utils.utils import clean_sentence
from src.settings.settings import Settings, Experiments_Settings
from src.components.models.embeddings.embeddings import (
    Langchain_OpenAI_Embeddings,
    Openai_Embeddings,
    HF_MultilingualE5_Embeddings,
)
from langchain_community.vectorstores.chroma import (
    Chroma as Langchain_Chroma_Collection,
)
from src.settings.env_config import Config
from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    HuggingFaceEmbeddingFunction,
)
import chromadb

import pandas as pd


class DataIndexerAssistant:
    def __init__(self) -> None:
        # Iniciando modelos
        print("Iniciando modelos")
        self.init_models()

        # Configuracion cliente chromadb
        self.chroma_dev_db_path = Settings.Chroma.get_db_path()
        self.chroma_experiments_db_path = Experiments_Settings.Chroma.get_db_path()

        self.chromadb_client = chromadb.PersistentClient(path=self.chroma_dev_db_path)
        
        # TODO PARA EXPERIMENTS
        # self.chromadb_experiments_client = chromadb.PersistentClient(
        #     path=self.chroma_experiments_db_path
        # )
        # TODO PARA EXPERIMENTS

        # Configuracion embeddings functions
        self.openai_embedding_function = OpenAIEmbeddingFunction(
            api_key=Settings.Openai.get_api_key(),
            model_name=Settings.Openai.get_embeddings_model(),
        )
        self.huggingface_embedding_function = HuggingFaceEmbeddingFunction(
            api_key=Settings.Hugging_face.get_api_key(),
            model_name=Settings.Hugging_face.get_embeddings_model_mle5(),
        )

        # Configuracion nombre de colecciones
        self.collection_names = {
            "SQL_EXAMPLES_COLLECTION": Config.get_chromadb_config()[
                "SQL_EXAMPLES_COLLECTION"
            ],
            "CLASSIFIER_COLLECTION": Config.get_chromadb_config()[
                "CLASSIFIER_COLLECTION"
            ],
            "TABLE_DEFINITIONS_COLLECTION": Config.get_chromadb_config()[
                "TABLE_DEFINITIONS_COLLECTION"
            ],
            "RELATIONS_DEFINITIONS_COLLECTION": Config.get_chromadb_config()[
                "RELATIONS_DEFINITIONS_COLLECTION"
            ],
            "COLUMNS_DEFINITIONS_COLLECTION": Config.get_chromadb_config()[
                "COLUMNS_DEFINITIONS_COLLECTION"
            ],
            # "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS": Config.get_experimentsdb_config()[
            #     "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"
            # ],
            # "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS": Config.get_experimentsdb_config()[
            #     "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"
            # ],
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
                "SQL_EXAMPLES_COLLECTION"
            ]: lambda: self._delete_collection(
                self.collection_names["SQL_EXAMPLES_COLLECTION"]
            ),
            self.collection_names[
                "CLASSIFIER_COLLECTION"
            ]: lambda: self._delete_collection(
                self.collection_names["CLASSIFIER_COLLECTION"]
            ),
            self.collection_names[
                "TABLE_DEFINITIONS_COLLECTION"
            ]: lambda: self._delete_collection(
                self.collection_names["TABLE_DEFINITIONS_COLLECTION"]
            ),
            self.collection_names[
                "RELATIONS_DEFINITIONS_COLLECTION"
            ]: lambda: self._delete_collection(
                self.collection_names["RELATIONS_DEFINITIONS_COLLECTION"]
            ),
            self.collection_names[
                "COLUMNS_DEFINITIONS_COLLECTION"
            ]: lambda: self._delete_collection(
                self.collection_names["COLUMNS_DEFINITIONS_COLLECTION"]
            ),
            # self.collection_names[
            #     "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"
            # ]: lambda: self._delete_collection(
            #     self.collection_names["EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"]
            # ),
            # self.collection_names[
            #     "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"
            # ]: lambda: self._delete_collection(
            #     self.collection_names["EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"]
            # ),
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
    # def train_with_chroma(
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

    def train_classifier_collection(self):
        # Leyendo el diccionario del excel
        excel_data = read_classify_dictionary()
        vectors = []
        metadatas = []
        ids = []
        for index, row in excel_data.iterrows():
            processed_word = clean_sentence(row["input"])
            new_embedding: list[float] = self.openai_native_model.get_embeddings(
                processed_word
            )
            # new_embedding: list[float] = get_embeddings(
            #     processed_word,
            #     self.openai_model,
            #     self.settings.openai.embeddings_model,
            # )
            vectors.append(new_embedding)
            metadatas.append(
                {
                    "input": row["input"],
                    "analysis": row["analysis"],
                    "response": row["response"],
                }
            )
            ids.append(f"id_sql_ex_col_{index}")
        self.index_with_vectors(
            collection_name=self.collection_names["CLASSIFIER_COLLECTION"],
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas,
            embedding_function=self.openai_embedding_function,
        )
        print("Colección classifier entrenada!")

    def train_sql_examples_collection(self):
        # Leyendo el diccionario del excel
        excel_data = read_sql_examples()
        vectors = []
        metadatas = []
        ids = []
        for index, row in excel_data.iterrows():
            processed_word = clean_sentence(row["questions"])
            new_embedding: list[float] = self.openai_native_model.get_embeddings(
                processed_word
            )
            # new_embedding: list[float] = get_embeddings(
            #     processed_word,
            #     self.openai_model,
            #     self.settings.openai.embeddings_model,
            # )
            vectors.append(new_embedding)
            metadatas.append(
                {
                    "question": row["questions"],
                    "answer": row["answers"],
                }
            )
            ids.append(f"id_sql_ex_col_{index}")
        self.index_with_vectors(
            collection_name=self.collection_names["SQL_EXAMPLES_COLLECTION"],
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas,
            embedding_function=self.openai_embedding_function,
        )
        print("Colección sql_examples entrenada!")

    def train_table_definitions_collection(self):
        # Leyendo el diccionario del excel
        excel_data = read_database_semantics(sheet_name="semantics_tables")
        vectors = []
        metadatas = []
        ids = []
        for index, row in excel_data.iterrows():
            processed_word = clean_sentence(row["semantic_table_description"])
            new_embedding: list[float] = self.openai_native_model.get_embeddings(
                processed_word
            )
            # new_embedding: list[float] = get_embeddings(
            #     # row["semantic_table_description"],
            #     processed_word,
            #     self.openai_model,
            #     self.settings.openai.embeddings_model,
            # )
            vectors.append(new_embedding)
            metadatas.append(
                {
                    "table_name": row["meta_table_name"],
                    "table_schema": row["meta_table_schema"],
                }
            )
            ids.append(f"id_table_definitions_col_{index}")
        self.index_with_vectors(
            collection_name=self.collection_names["TABLE_DEFINITIONS_COLLECTION"],
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas,
            embedding_function=self.openai_embedding_function,
        )
        print("Colección table_definitions entrenada!")

    def train_relations_definitions_collection(self):
        # Leyendo el diccionario del excel
        excel_df = read_database_semantics(
            sheet_name="semantics_relations",
            cols=[
                "semantic_table_relation",
                "meta_table_1",
                "meta_table_2",
                "meta_k1",
                "meta_k2",
                "meta_mid_table",
                "meta_mid_k1",
                "meta_mid_k2",
                "meta_relation_description",
                "meta_mid_k2_description",
                "meta_mid_k1_description",
                "meta_k2_description",
                "meta_k1_description",
            ],
        )
        vectors = []
        metadatas = []
        ids = []
        for index, row in excel_df.iterrows():
            if not pd.isna(row["semantic_table_relation"]):
                # processed_word = clean_sentence(row["semantic_table_relation"])
                new_embedding: list[float] = self.openai_native_model.get_embeddings(
                    row["semantic_table_relation"]
                )
                # new_embedding: list[float] = get_embeddings(
                #     row["semantic_table_relation"],
                #     # processed_word,
                #     self.openai_model,
                #     self.settings.openai.embeddings_model,
                # )
                vectors.append(new_embedding)
                metadatas.append(
                    {
                        "table_1": row["meta_table_1"],
                        "table_2": row["meta_table_2"],
                        "key_table_1": row["meta_k1"],
                        "key_table_2": row["meta_k2"],
                        "mid_table": (
                            ""
                            if pd.isna(row["meta_mid_table"])
                            else row["meta_mid_table"]
                        ),
                        "key_mid_table_1": (
                            "" if pd.isna(row["meta_mid_k1"]) else row["meta_mid_k1"]
                        ),
                        "key_mid_table_2": (
                            "" if pd.isna(row["meta_mid_k2"]) else row["meta_mid_k2"]
                        ),
                        "relation_description": (
                            ""
                            if pd.isna(row["meta_relation_description"])
                            else row["meta_relation_description"]
                        ),
                        "key_mid_table_1_description": (
                            ""
                            if pd.isna(row["meta_mid_k1_description"])
                            else row["meta_mid_k1_description"]
                        ),
                        "key_mid_table_2_description": (
                            ""
                            if pd.isna(row["meta_mid_k2_description"])
                            else row["meta_mid_k2_description"]
                        ),
                        "key_table_1_description": (
                            ""
                            if pd.isna(row["meta_k1_description"])
                            else row["meta_k1_description"]
                        ),
                        "key_table_2_description": (
                            ""
                            if pd.isna(row["meta_k2_description"])
                            else row["meta_k2_description"]
                        ),
                    }
                )
                ids.append(f"id_relations_definitions_col_{index}")
        self.index_with_vectors(
            collection_name=self.collection_names["RELATIONS_DEFINITIONS_COLLECTION"],
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas,
            embedding_function=self.openai_embedding_function,
        )
        print("Colección relations_definitions entrenada!")

    def train_columns_definitions_collection(self):
        # Leyendo el diccionario del excel
        excel_df = read_database_semantics(
            sheet_name="semantics_columns",
            cols=[
                "semantic_column",
                "meta_table",
                "meta_column_name",
                "meta_column_type",
                "meta_column_comment",
                "meta_priority",
            ],
        )
        vectors = []
        metadatas = []
        ids = []
        for index, row in excel_df.iterrows():
            if not pd.isna(row["semantic_column"]):
                # processed_word = clean_sentence(row["semantic_table_relation"])
                new_embedding: list[float] = self.openai_native_model.get_embeddings(
                    row["semantic_column"]
                )
                # new_embedding: list[float] = get_embeddings(
                #     row["semantic_column"],
                #     # processed_word,
                #     self.openai_model,
                #     self.settings.openai.embeddings_model,
                # )
                vectors.append(new_embedding)
                metadatas.append(
                    {
                        "meta_table": (
                            "" if pd.isna(row["meta_table"]) else row["meta_table"]
                        ),
                        "meta_column_name": (
                            ""
                            if pd.isna(row["meta_column_name"])
                            else row["meta_column_name"]
                        ),
                        "meta_column_type": (
                            ""
                            if pd.isna(row["meta_column_type"])
                            else row["meta_column_type"]
                        ),
                        "meta_priority": (
                            ""
                            if pd.isna(row["meta_priority"])
                            else row["meta_priority"]
                        ),
                        "meta_column_comment": (
                            ""
                            if pd.isna(row["meta_column_comment"])
                            else row["meta_column_comment"]
                        ),
                    }
                )
                ids.append(f"id_columns_definitions_col_{index}")
        self.index_with_vectors(
            collection_name=self.collection_names["COLUMNS_DEFINITIONS_COLLECTION"],
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas,
            embedding_function=self.openai_embedding_function,
        )
        print("Colección columns_definitions entrenada!")

    # TODO PARA EXPERIMENTS

    # def train_openai_experiments_collection(self):
    #     # Leyendo el diccionario del excel
    #     excel_data = read_database_experiments(
    #         sheet_name="bussiness_semantics",
    #         cols=[
    #             "semantic_term_description",
    #             "meta_term",
    #             "meta_table_name",
    #             "meta_terms_definitions",
    #             "meta_terms_replacements",
    #             "meta_sql_advices",
    #         ],
    #     )
    #     vectors = []
    #     metadatas = []
    #     ids = []
    #     for index, row in excel_data.iterrows():
    #         processed_word = clean_technical_term(row["semantic_term_description"])
    #         new_embedding: list[float] = self.openai_native_model.get_embeddings(processed_word)
    #         # new_embedding: list[float] = get_embeddings(
    #         #     processed_word,
    #         #     self.openai_model,
    #         #     self.settings.openai.embeddings_model,
    #         # )
    #         # new_embedding = get_hf_embeddings(processed_word)
    #         vectors.append(new_embedding)
    #         metadatas.append(
    #             {
    #                 "meta_term": row["meta_term"],
    #                 "meta_table_name": row["meta_table_name"],
    #                 "meta_terms_definitions": (
    #                     " "
    #                     if pd.isna(row["meta_terms_definitions"])
    #                     else row["meta_terms_definitions"]
    #                 ),
    #                 "meta_terms_replacements": (
    #                     " "
    #                     if pd.isna(row["meta_terms_replacements"])
    #                     else row["meta_terms_replacements"]
    #                 ),
    #             }
    #         )
    #         ids.append(f"id_experiments_col_{index}")
    #     self.index_with_vectors(
    #         collection_name=self.collection_names[
    #             "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"
    #         ],
    #         embeddings=vectors,
    #         ids=ids,
    #         metadatas=metadatas,
    #         embedding_function=self.openai_embedding_function
    #     )
    #     print("Colección openai_experiments entrenada!")

    # def train_llama_experiments_collection(self):
    #     # Leyendo el diccionario del excel
    #     excel_data = read_database_experiments(
    #         sheet_name="bussiness_semantics",
    #         cols=[
    #             "semantic_term_description",
    #             "meta_term",
    #             "meta_table_name",
    #             "meta_terms_definitions",
    #             "meta_terms_replacements",
    #             "meta_sql_advices",
    #         ],
    #     )
    #     documents = []
    #     metadatas = []
    #     ids = []
    #     for index, row in excel_data.iterrows():
    #         processed_word = clean_technical_term(row["semantic_term_description"])
    #         # new_embedding: list[float] = get_embeddings(
    #         #     processed_word,
    #         #     self.openai_model,
    #         #     self.settings.openai.embeddings_model,
    #         # )
    #         # new_embedding = get_hf_embeddings(processed_word)
    #         documents.append(processed_word)
    #         metadatas.append(
    #             {
    #                 "meta_term": row["meta_term"],
    #                 "meta_table_name": row["meta_table_name"],
    #                 "meta_terms_definitions": (
    #                     " "
    #                     if pd.isna(row["meta_terms_definitions"])
    #                     else row["meta_terms_definitions"]
    #                 ),
    #                 "meta_terms_replacements": (
    #                     " "
    #                     if pd.isna(row["meta_terms_replacements"])
    #                     else row["meta_terms_replacements"]
    #                 ),
    #             }
    #         )
    #         ids.append(f"id_experiments_llama_col_{index}")
    #     self.index_with_documents(
    #         collection_name=self.collection_names[
    #             "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"
    #         ],
    #         ids=ids,
    #         embedding_function=self.huggingface_embedding_function,
    #         documents=documents,
    #         metadatas=metadatas,
    #     )
    #     print("Colección llama_experiments entrenada!")
    
    # TODO PARA EXPERIMENTS

    def train_specific_collection(self):
        collection_names = {
            self.collection_names[
                "SQL_EXAMPLES_COLLECTION"
            ]: lambda: self.train_sql_examples_collection(),
            self.collection_names[
                "CLASSIFIER_COLLECTION"
            ]: lambda: self.train_classifier_collection(),
            self.collection_names[
                "TABLE_DEFINITIONS_COLLECTION"
            ]: lambda: self.train_table_definitions_collection(),
            self.collection_names[
                "RELATIONS_DEFINITIONS_COLLECTION"
            ]: lambda: self.train_relations_definitions_collection(),
            self.collection_names[
                "COLUMNS_DEFINITIONS_COLLECTION"
            ]: lambda: self.train_columns_definitions_collection(),
            # self.collection_names[
            #     "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"
            # ]: lambda: self.train_openai_experiments_collection(),
            # self.collection_names[
            #     "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"
            # ]: lambda: self.train_llama_experiments_collection(),
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
