import environ

env = environ.Env()
environ.Env.read_env()

class Config:
    @staticmethod
    def get_openai_config():
        return {
            "OPENAI_API_KEY": env("OPENAI_API_KEY"),
            "OPENAI_LLM_MODEL": env("OPENAI_LLM_MODEL"),
            "OPENAI_EMBEDDINGS_MODEL": env("OPENAI_EMBEDDINGS_MODEL"),
        }

    @staticmethod
    def get_chromadb_config():
        return {
            "CHROMADB_DIRECTORY": env("CHROMADB_DIRECTORY"),
            
            "KEYWORDS_COLLECTION": env("KEYWORDS_COLLECTION"),
            "SUMMARY_COLLECTION": env("SUMMARY_COLLECTION"),
            "CLASSIFIER_COLLECTION": env("CLASSIFIER_COLLECTION"),
            "CONTEXT_COLLECTION": env("CONTEXT_COLLECTION"),
            "SQL_EXAMPLES_COLLECTION": env("SQL_EXAMPLES_COLLECTION"),
            "TABLE_DEFINITIONS_COLLECTION": env("TABLE_DEFINITIONS_COLLECTION"),
            "COLUMNS_DEFINITIONS_COLLECTION": env("COLUMNS_DEFINITIONS_COLLECTION"),
            "RELATIONS_DEFINITIONS_COLLECTION": env("RELATIONS_DEFINITIONS_COLLECTION"),
        }
        
    @staticmethod
    def get_experimentsdb_config():
        return {
            "CHROMADB_EXPERIMENTS_DIRECTORY": env("CHROMADB_EXPERIMENTS_DIRECTORY"),
            
            "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS": env("EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"),
            "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS": env("EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"),
        }

    @staticmethod
    def get_sqldatabase_config():
        return {
            "DB_USER": env("USER"),
            "DB_PWD": env("PWD"),
            "DB_HOST": env("SERVER"),
            "DB_NAME": env("DBNAME"),
            "DB_DRIVER": env("ODBCDRIVER"),
        }

    @staticmethod
    def get_hf_config():
        return {
            "HF_KEY": env("HF_KEY"),
            "HF_META_LLAMA_LLAMA38B_MODEL": env("HF_META_LLAMA_LLAMA38B_MODEL"),
            "HF_INFLOAT_MLE5_EMBEDDINGS_MODEL": env("HF_INFLOAT_MLE5_EMBEDDINGS_MODEL"),
        }
