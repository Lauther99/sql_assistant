import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3")
from langchain_openai import OpenAI as OpenAI_From_Langchain
from src.settings.settings import Settings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI as OpenAI_From_OpenAILibrary
from abc import ABC, abstractmethod
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import requests
import time

general_chain_template: str = (
    """Your name is M-Assistant, very helpful assistant expert in measurement systems.\n{task}"""
)


class Model(ABC):
    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def query_llm(self, input):
        pass

    @abstractmethod
    def get_embeddings(self, input):
        pass

# Modelo de Openai llamado con Langchain
class Langchain_Model(Model):
    def __init__(self) -> None:
        self.langchain_llm = OpenAI_From_Langchain(
            model=Settings.Openai.get_llm_model_name(),
            temperature=0,
            api_key=Settings.Openai.get_api_key(),
            verbose=True,
            max_tokens=-1,
        )
        self.chain = None

    def init_model(self):
        self.chain = LLMChain(
            llm=self.langchain_llm,
            verbose=False,
            prompt=PromptTemplate.from_template(general_chain_template),
        )

    def query_llm(self, input):
        response = self.chain.invoke(input={"task": input})
        return response

    def get_embeddings(self, input):
        embeddings = OpenAIEmbeddings(
            model=Settings.Openai.get_embeddings_model(),
            openai_api_key=Settings.Openai.get_api_key(),
        )
        query_result = embeddings.embed_query(input)
        return query_result

# Modelo Openai llamado directamente
class Openai_Model(Model):
    def init_model(self):
        self.openai_endpoint = OpenAI_From_OpenAILibrary(
            api_key=Settings.Openai.get_api_key()
        )

    def query_llm(self, input):
        response = self.openai_endpoint.completions.create(
            model=Settings.Openai.get_llm_model_name(),
            prompt=input,
            temperature=0
        )
        return response.choices[0].text

    def get_embeddings(self, input):
        embeddings = self.openai_endpoint.embeddings.create(
            input=input, model=Settings.Openai.get_embeddings_model()
        )
        return embeddings.data[0].embedding

# Modelo mixto Llama3(llm) + MultiLingualE5(embeddings)
class HuggingFace_Model(Model):
    def init_model(self):
        self.llm_endpoint = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
        self.embedding_endpoint = "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-large-instruct"
        self.headers = {
            "Authorization": f"""Bearer {Settings.Hugging_face.get_api_key()}"""
        }

    def query_llm(self, input):
        json_payload = {
            "inputs": input,
        }
        response = requests.post(self.llm_endpoint, headers=self.headers, json=json_payload).json()
        
        return str(response[0]["generated_text"]).split("assistant<|end_header_id|>")[1]

    def get_embeddings(self, input):
        json_payload = {
            "inputs": input,
        }
        vector = requests.post(self.embedding_endpoint, headers=self.headers, json=json_payload).json()
        
        if not isinstance(vector, list) and not all(isinstance(i, float) for i in vector):
            print("Se esta levantando el modelo de embeddings :D")
            estimated_time = float(vector["estimated_time"])
            time.sleep(estimated_time + 10)
            vector = self.get_embeddings(input)
        
        return vector
