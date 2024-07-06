from abc import ABC, abstractmethod


class Base_LLM_Response(ABC):
    def __init__(self, type, prompt, llm_response, json_response):
        self.type = type
        self.prompt = prompt
        self.llm_response = llm_response
        self.json_response = json_response

    def __repr__(self):
        return f"Base_LLM_Response(type={self.type}, prompt={f'''{self.prompt[:40]}...'''}, llm_response={self.llm_response}, json_response={self.json_response})"


class Base_LLM(ABC):
    def __init__(self):
        self.responses: list[Base_LLM_Response] = []

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def query_llm(self, input):
        pass
    
    @abstractmethod
    def apply_model_template(self, instruction, suffix):
        pass

    def add_response(self, type, prompt, llm_response, json_response):
        response = Base_LLM_Response(type, prompt, llm_response, json_response)
        self.responses.append(response)

    def get_responses(self):
        return self.responses


class Base_Embeddings(ABC):
    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def get_embeddings(self, input):
        pass
