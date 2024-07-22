import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3")
from typing import Any, Hashable, Optional
import pandas as pd
from abc import ABC
import uuid
from datetime import datetime


class Base_Message(ABC):
    def __init__(self, message="", message_type=""):
        self.message_id = uuid.uuid4()
        self.message = message
        self.message_type = message_type
        self.date_created = datetime.now()


class AIMessage(Base_Message):
    def __init__(
        self,
        user_message_id: uuid,
        message="",
        message_type="",
        dataframe=None,
        sql_response: str | None = "",
    ) -> None:
        super().__init__(message, message_type)
        self.replied_user_message_id = user_message_id
        self.sql_response = sql_response
        self.dataframe: list[dict[Hashable, Any]] = dataframe

    def __repr__(self):
        return (
            f"AIMessage(\n"
            f"  replied_user_message_id={self.replied_user_message_id!r}\n"
            f"  message_id={self.message_id!r},\n"
            f"  message={self.message!r},\n"
            f"  message_type={self.message_type!r},\n"
            f"  date_created={self.date_created!r},\n"
            f"  sql_response={self.sql_response!r},\n"
            f"  dataframe={self.dataframe if self.dataframe is None else 'DataFrame(...)'},\n"
            f")"
        )


class HumanMessage(Base_Message):
    def __init__(self, message="", message_type="") -> None:
        super().__init__(message, message_type)

    def __repr__(self):
        return (
            f"HumanMessage(\n"
            f"  message_id={self.message_id!r},\n"
            f"  message={self.message!r},\n"
            f"  message_type={self.message_type!r},\n"
            f"  date_created={self.date_created!r}\n"
            f")"
        )
