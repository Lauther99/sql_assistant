
from langchain_openai import OpenAI as OpenAI_From_Langchain
from src.components.models.models_interfaces import Base_LLM
from src.settings.settings import Settings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from openai import OpenAI as OpenAI_From_OpenAILibrary
import requests

# Modelo de Openai llamado con Langchain
class Langchain_OpenAI_LLM(Base_LLM):
    def __init__(self) -> None:
        super().__init__()
        self.langchain_llm = OpenAI_From_Langchain(
            model=Settings.Openai.get_llm_model_name(),
            temperature=0,
            api_key=Settings.Openai.get_api_key(),
            verbose=True,
            max_tokens=-1,
        )
        self.chain = None
        self.general_chain_template = """Your name is M-Assistant, very helpful assistant expert in measurement systems.\n{task}"""

    def init_model(self):
        self.chain = LLMChain(
            llm=self.langchain_llm,
            verbose=False,
            prompt=PromptTemplate.from_template(self.general_chain_template),
        )

    def query_llm(self, input):
        response = self.chain.invoke(input={"task": input})
        return response


# Modelo Openai llamado directamente
class Openai_LLM(Base_LLM):
    def __init__(self) -> None:
        super().__init__()
        
    def init_model(self):
        self.openai_endpoint = OpenAI_From_OpenAILibrary(
            api_key=Settings.Openai.get_api_key()
        )

    def query_llm(self, input):
        response = self.openai_endpoint.completions.create(
            model=Settings.Openai.get_llm_model_name(), prompt=input, temperature=0
        )
        return response.choices[0].text


# Modelo mixto Llama3(llm) + MultiLingualE5(embeddings)
class HF_Llama38b_LLM(Base_LLM):
    def __init__(self) -> None:
        super().__init__()
        self.instruction_suffix = ""
    
    def init_model(self):
        self.llm_endpoint = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
        self.headers = {
            "Authorization": f"""Bearer {Settings.Hugging_face.get_api_key()}"""
        }

    def query_llm(self, input):
        json_payload = {
            "inputs": input,
        }
        response = requests.post(
            self.llm_endpoint, headers=self.headers, json=json_payload
        ).json()

        return str(response[0]["generated_text"]).split("assistant<|end_header_id|>")[1]

    def apply_chat_template(self, instruction, instruction_suffix):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Your are a very helpfull assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>
assistant<|end_header_id|>
{instruction_suffix}"""


