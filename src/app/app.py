from abc import ABC, abstractmethod
from typing import Optional

class AppDataCollector(ABC):
    def __init__(self):
        self.user_request: Optional[str] = None
        self.modified_user_request: Optional[str] = None
        self.request_type: Optional[str] = None
        self.simple_response: Optional[str] = None
        self.technical_terms: Optional[list[str]] = None
        self.terms_dictionary: Optional[list[dict[str, any]]] = None
        self.flavored_request_for_semantic_search: Optional[str] = None
        self.semantic_info: Optional[dict[str, any]] = None
        

    # def __repr__(self):
    #     return f"Base_LLM_Response(type={self.type}, prompt={f'''{self.prompt[:40]}...'''}, llm_response={self.llm_response}, json_response={self.json_response})"
