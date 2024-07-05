from abc import ABC, abstractmethod

class LLMResponse(ABC):
    def __init__(self, key):
        self.key = key
        self.llm_prompt = None
        self.llm_response = None
        self.llm_json_response = None

    def set_llm_prompt(self, input):
        self.llm_prompt = input
        
    def set_llm_response(self, input):
        self.llm_response = input
    
    def set_llm_json_response(self, input):
        self.llm_json_response = input
        
    def get_llm_prompt(self):
        return self.llm_prompt 

    def get_llm_response(self):
        return self.llm_response
        
    def get_llm_json_response(self):
        return self.llm_json_response 
    

class Request_Response(LLMResponse):
    def __init__(self):
        super().__init__("request")

class Request_Type_Response(LLMResponse):
    def __init__(self):
        super().__init__("type")
        
class Greeting_Response(LLMResponse):
    def __init__(self):
        super().__init__("greeting_reponse")

class Tables_Selected_Response(LLMResponse):
    def __init__(self):
        super().__init__("select_tables")

class Generate_SQL_Response(LLMResponse):
    def __init__(self):
        super().__init__("generate_sql")

class SQL_Classifier_Response(LLMResponse):
    def __init__(self):
        super().__init__("sql_classifier")

class SQL_Pre_Query(LLMResponse):
    def __init__(self):
        super().__init__("sql_pre_query")

class SQL_Summary_Response(LLMResponse):
    def __init__(self):
        super().__init__("sql_summary_response")
        
class Translate_Response(LLMResponse):
    def __init__(self):
        super().__init__("translated_response")
        
# MANAGER RESPONSES

class LLMResponseManager:
    def __init__(self):
        self.responses: list[LLMResponse] = []
        
    def add_response(self, response: LLMResponse):
        if not isinstance(response, LLMResponse):
            raise TypeError("response must be an instance of LLMResponse")
        self.responses.append(response)

    def display_responses(self):
        for response in self.responses:
            print(f"Class: {response.__class__.__name__}")
            print(f"  LLM Response: {response.get_llm_response()}")
            print(f"  LLM Response Parsed: {response.get_llm_json_response()}")
            print("-" * 40)