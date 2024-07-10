from abc import ABC, abstractmethod

class Base_LLM(ABC):
    # def __init__(self):
    #     self.responses: list[Base_LLM_Response] = []

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def query_llm(self, input):
        pass
    
    @abstractmethod
    def apply_model_template(self, instruction, suffix):
        pass


class Base_Embeddings(ABC):
    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def get_embeddings(self, input):
        pass
