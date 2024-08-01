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
    def get_experimentsdb_config():
        return {
            "CHROMADB_EXPERIMENTS_DIRECTORY": env("CHROMADB_EXPERIMENTS_DIRECTORY"),
            "EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS": env("EXPERIMENTS_COLLECTION_OPENAI_EMBEDDINGS"),
            "EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS": env("EXPERIMENTS_COLLECTION_LLAMA_EMBEDDINGS"),
            "EXPERIMENTS_SEMANTIC_TABLES": env("EXPERIMENTS_SEMANTIC_TABLES"),
            "EXPERIMENTS_SEMANTIC_RELATIONS": env("EXPERIMENTS_SEMANTIC_RELATIONS"),
            "EXPERIMENTS_COLUMNS": env("EXPERIMENTS_COLUMNS"),
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
