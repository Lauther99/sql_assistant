from langchain_openai import OpenAIEmbeddings
from src.components.models.models_interfaces import Base_Embeddings
from src.settings.settings import Settings
from openai import OpenAI as OpenAI_From_OpenAILibrary
import requests
import time


class Langchain_OpenAI_Embeddings(Base_Embeddings):
    def __init__(self) -> None:
        super().__init__()
        
    def init_model(self):
        self.embeddings = OpenAIEmbeddings(
            model=Settings.Openai.get_embeddings_model(),
            openai_api_key=Settings.Openai.get_api_key(),
        )

    def get_embeddings(self, input):
        query_result = self.embeddings.embed_query(input)
        return query_result


class Openai_Embeddings(Base_Embeddings):
    def __init__(self) -> None:
        super().__init__()
        
    def init_model(self):
        self.openai_endpoint = OpenAI_From_OpenAILibrary(
            api_key=Settings.Openai.get_api_key()
        )

    def get_embeddings(self, input):
        embeddings = self.openai_endpoint.embeddings.create(
            input=input, model=Settings.Openai.get_embeddings_model()
        )
        return embeddings.data[0].embedding


class HF_MultilingualE5_Embeddings(Base_Embeddings):
    def __init__(self) -> None:
        super().__init__()
        
    def init_model(self):
        self.embedding_endpoint = "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-large-instruct"

        self.headers = {
            "Authorization": f"""Bearer {Settings.Hugging_face.get_api_key()}"""
        }

    def get_embeddings(self, input):
        json_payload = {
            "inputs": input,
        }
        vector = requests.post(
            self.embedding_endpoint, headers=self.headers, json=json_payload
        ).json()

        if not isinstance(vector, list) and not all(
            isinstance(i, float) for i in vector
        ):
            estimated_time = float(vector["estimated_time"])
            print(f"Se esta levantando el modelo de embeddings :D, listo en: {estimated_time}s")
            time.sleep(estimated_time + 10)
            vector = self.get_embeddings(input)

        return vector
