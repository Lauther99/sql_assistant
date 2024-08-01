from abc import ABC, abstractmethod
from typing import Any, Hashable, Optional
import pandas as pd
from src.components.memory.memory_interfaces import HumanMessage, AIMessage


class ConversationData(ABC):
    def __init__(self):
        self.current_conversation_summary: Optional[str] = None
        self.current_slots: Optional[str] = None
        self.last_user_message: Optional[HumanMessage] = None
        self.last_ai_message: Optional[AIMessage] = None

    def __repr__(self):
        return (
            f"ConversationData(\n"
            f"  current_conversation_summary={self.current_conversation_summary!r},\n"
            f"  current_slots={self.current_slots!r},\n"
            f"  last_user_message={self.last_user_message!r},\n"
            f"  last_ai_message={self.last_ai_message!r}\n"
            f")"
        )



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
        self.dataframe_response: list[dict[Hashable, Any]] = None

        self.ai_post_response: Optional[str] = None
        self.detected_language: Optional[str] = None
        self.semantic_list_terms: Optional[list[str]] = None

    def __repr__(self):
        return (
            f"AppDataCollector(\n"
            f"  current_conversation_data={self.current_conversation_data!r},\n"
            f"  conversation_summary={self.conversation_summary!r},\n"
            f"  conversation_slots={self.conversation_slots!r},\n"
            f"  user_request={self.user_request!r},\n"
            f"  modified_user_request={self.modified_user_request!r},\n"
            f"  request_type={self.request_type!r},\n"
            f"  simple_response={self.simple_response!r},\n"
            f"  technical_terms={self.technical_terms!r},\n"
            f"  terms_dictionary={self.terms_dictionary!r},\n"
            f"  flavored_request_for_semantic_search={self.flavored_request_for_semantic_search!r},\n"
            f"  semantic_info={self.semantic_info!r},\n"
            f"  sql_code={self.sql_code!r},\n"
            f"  assistant_sql_code_class={self.assistant_sql_code_class!r},\n"
            f"  assistant_sql_code_analysis={self.assistant_sql_code_analysis!r},\n"
            f"  assistant_sql_code_suggestion={self.assistant_sql_code_suggestion!r},\n"
            f"  sql_pre_query={self.sql_pre_query!r},\n"
            f"  ai_pre_response={self.ai_pre_response!r},\n"
            f"  dataframe_response={self.dataframe_response if self.dataframe_response is None else 'DataFrame(...)'},\n"
            f"  ai_post_response={self.ai_post_response!r},\n"
            f"  detected_language={self.detected_language!r}\n"
            f")"
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

    def add_response(
        self, type: str, prompt: str, llm_response: str, json_response: str
    ):
        base_response = Base_LLM_Response(type, prompt, llm_response, json_response)
        self.llm_responses.append(base_response)

    def __repr__(self):
        return f"LLMResponseCollector(llm_responses={self.llm_responses})"
