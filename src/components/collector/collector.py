from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from src.components.memory.memory_interfaces import HumanMessage, AIMessage

class ConversationData(ABC):
    def __init__(self):
        self.current_conversation_summary: Optional[str] = None 
        self.current_slots: Optional[str] = None 
        self.last_user_message: Optional[HumanMessage] = None 
        self.last_ai_message: Optional[AIMessage] = None 
        

class AppDataCollector(ABC):
    def __init__(self):
        self.current_conversation_data: Optional[ConversationData] = ConversationData()
        
        self.conversation_summary: Optional[str] = None
        self.conversation_slots: Optional[str] = None
        self.user_request: Optional[str] = None

        self.modified_user_request: Optional[str] = None
        self.request_type: Optional[str] = None
        self.simple_response: Optional[str] = None
        self.technical_terms: Optional[list[str]] = None
        self.terms_dictionary: Optional[list[dict[str, any]]] = None
        self.flavored_request_for_semantic_search: Optional[str] = None
        self.semantic_info: Optional[dict[str, any]] = None

        self.sql_code: Optional[str] = None

        self.assistant_sql_code_class: Optional[str] = None
        self.assistant_sql_code_analysis: Optional[str] = None
        self.assistant_sql_code_suggestion: Optional[str] = None

        self.sql_pre_query: Optional[str] = None
        self.ai_pre_response: Optional[str] = None
        self.dataframe_response: Optional[pd.DataFrame] = None

        self.ai_post_response: Optional[str] = None
        self.detected_language: Optional[str] = None
        

    def __repr__(self):
        return (
            f"AppDataCollector(user_request={self.user_request!r}, modified_user_request={self.modified_user_request!r}, "
            f"request_type={self.request_type!r}, simple_response={self.simple_response!r}, technical_terms={self.technical_terms!r}, "
            f"terms_dictionary={self.terms_dictionary!r}, flavored_request_for_semantic_search={self.flavored_request_for_semantic_search!r}, "
            f"semantic_info={self.semantic_info!r}, sql_code={self.sql_code!r}, assistant_sql_code_class={self.assistant_sql_code_class!r}, "
            f"assistant_sql_code_analysis={self.assistant_sql_code_analysis!r}, assistant_sql_code_suggestion={self.assistant_sql_code_suggestion!r}, "
            f"sql_pre_query={self.sql_pre_query!r}, ai_pre_response={self.ai_pre_response!r}, dataframe_response={self.dataframe_response!r}, "
            f"ai_post_response={self.ai_post_response!r}, detected_language={self.detected_language!r})"
        )

class Base_LLM_Response(ABC):
    def __init__(self, type: str, prompt: str, llm_response: str, json_response: str):
        self.type = type
        self.prompt = prompt
        self.llm_response = llm_response
        self.json_response = json_response

    def __repr__(self):
        return f"Base_LLM_Response(type={self.type}, prompt={f'''{self.prompt[:40]}...'''}, llm_response={self.llm_response}, json_response={self.json_response})"
    
class LLMResponseCollector(ABC):
    def __init__(self):
        self.llm_responses: list[Base_LLM_Response] = []

    def add_response(self, type: str, prompt: str, llm_response: str, json_response: str):
        base_response = Base_LLM_Response(type, prompt, llm_response, json_response)
        self.llm_responses.append(base_response)
        
    def __repr__(self):
        return f"LLMResponseCollector(llm_responses={self.llm_responses})"