import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3")
from typing import Any, Hashable, List, Optional, Union
import pandas as pd
from abc import ABC
import uuid
from datetime import datetime
from src.components.memory.memory_interfaces import (
    AIMessage,
    HumanMessage,
    Base_Message,
)
import copy


class ChatDocument(ABC):
    def __init__(self):
        self.user_id: uuid.UUID = None
        self.conversation_id: uuid.UUID | None = None

        self.messages: List[Union[AIMessage, HumanMessage]] = []
        self.last_user_message: HumanMessage | None = None
        self.last_assistant_message: AIMessage | None = None
        self.conversation_slots: str = None
        self.last_interaction: datetime | None = None
        self.current_summary: str | None = None

    def chat_document_to_dict(self) -> dict:
        return {
            "user_id": str(self.user_id) if self.conversation_id else None,
            "conversation_id": (
                str(self.conversation_id) if self.conversation_id else None
            ),
            "messages": [
                message.to_dict()
                for message in self.messages
                if hasattr(message, "to_dict")
            ],
            "last_user_message": (
                self.last_user_message.to_dict() if self.last_user_message else None
            ),
            "last_assistant_message": (
                self.last_assistant_message.to_dict()
                if self.last_assistant_message
                else None
            ),
            "conversation_slots": self.conversation_slots,
            "last_interaction": (
                self.last_interaction.isoformat() if isinstance(self.last_interaction, datetime) else datetime.now().isoformat()
            ),
            "current_summary": self.current_summary,
        }

    @classmethod
    def parse_dict_to_document(cls, data: dict) -> "ChatDocument":
        document = cls()
        document.user_id = (
            uuid.UUID(data.get("user_id")) if data.get("user_id") else None
        )
        document.conversation_id = (
            uuid.UUID(data.get("conversation_id"))
            if data.get("conversation_id")
            else None
        )
        document.messages = [
            cls.message_factory(m, AIMessage if m["message_type"] == "AI" else HumanMessage)
            for m in data.get("messages", [])
        ]
        document.last_user_message = cls.message_factory(
            data.get("last_user_message"), HumanMessage
        )
        document.last_assistant_message = cls.message_factory(
            data.get("last_assistant_message"), AIMessage
        )
        document.conversation_slots = data.get("conversation_slots")
        document.last_interaction = (
            datetime.fromisoformat(data.get("last_interaction")).isoformat()
            if data.get("last_interaction")
            else None
        )
        document.current_summary = data.get("current_summary")
        return document

    @staticmethod
    def message_factory(data, message_type: HumanMessage | AIMessage | None = None):
        if data is None:
            return None
        if message_type is None:
            # Assuming a general message class exists or use a default
            return Base_Message.from_dict(data)
        return message_type.from_dict(data)

    def __repr__(self):
        return (
            f"ChatDocument(user_id={self.user_id},\n"
            f"conversation_id={self.conversation_id},\n"
            f"messages={self.messages},\n"
            f"last_user_message={self.last_user_message},\n"
            f"last_assistant_message={self.last_assistant_message},\n"
            f"conversation_slots={self.conversation_slots},\n"
            f"last_interaction={self.last_interaction.isoformat()},\n"
            f"current_summary={self.current_summary})"
        )
        
    def copy(self) -> "ChatDocument":
            return copy.deepcopy(self)