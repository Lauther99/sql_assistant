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

    def apply_model_template(self, instruction, suffix):
        prompt = f"{instruction}\n\n{suffix}"
        return prompt


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

    def apply_model_template(self, instruction, suffix):
        prefix = "Your name is M-Assistant, very helpful assistant expert in measurement systems.\n"
        prompt = f"{prefix}\n\n{instruction}\n\n{suffix}"
        return prompt


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
        response = {}
        json_payload = {
            "inputs": input,
        }

        try:
            # Realiza la solicitud
            http_response = requests.post(
                self.llm_endpoint, headers=self.headers, json=json_payload
            )
            # Verifica el código de estado HTTP
            if http_response.status_code != 200:
                print(f"Error: Received status code {http_response.status_code}")
                return response

            # Intenta decodificar la respuesta como JSON
            try:
                output = http_response.json()
            except requests.exceptions.JSONDecodeError:
                print("Error: La respuesta no es un JSON válido.")
                print(http_response.text)
                return response

            # Procesa la respuesta JSON
            try:
                response["text"] = str(output[0]["generated_text"]).split("assistant<|end_header_id|>")[1]
            except (IndexError, KeyError) as e:
                print(f"Error al procesar la respuesta JSON: {e}")
                print(output)
                return response

        except requests.exceptions.RequestException as e:
            print(f"Error en la solicitud: {e}")
        
        return response

    def apply_model_template(self, instruction, suffix):
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYour are a very helpfull assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>\nassistant<|end_header_id|>\n{suffix}"""

        return prompt
