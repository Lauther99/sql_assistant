import sys


sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3")
from src.components.memory import MEMORY_TYPES
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
        
    def to_dict(self) -> dict:
        return {
            "message_id": str(self.message_id),
            "message": self.message,
            "message_type": self.message_type,
            "date_created": self.date_created.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict):
        data = data.copy()  # Evitar modificar el diccionario original

        # Manejar la conversión de tipos si es necesario
        if 'message_id' in data:
            data['message_id'] = uuid.UUID(data['message_id'])
        if 'date_created' in data:
            data['date_created'] = datetime.fromisoformat(data['date_created']).isoformat()

        # Pasar solo las claves que el constructor espera
        field_names = cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]
        init_args = {key: data[key] for key in field_names if key in data}
        return cls(**init_args)
    
    



class AIMessage(Base_Message):
    def __init__(
        self,
        replied_user_message_id: uuid.UUID,
        message="",
        message_type=MEMORY_TYPES["AI"],
        dataframe=None,
        sql_response: str | None = "",
    ) -> None:
        super().__init__(message, message_type)
        self.replied_user_message_id = replied_user_message_id
        self.sql_response = sql_response
        self.dataframe: list[dict[Hashable, Any]] = dataframe
        
    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict.update({
            "replied_user_message_id": str(self.replied_user_message_id),
            "sql_response": self.sql_response,
            "dataframe": self.dataframe,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data):
        data = data.copy()

        if 'replied_user_message_id' in data:
            data['replied_user_message_id'] = uuid.UUID(data['replied_user_message_id'])
        
        # Llamar al método de la clase base para manejar los campos comunes
        instance = super().from_dict(data)

        # Asignar campos adicionales
        instance.sql_response = data.get('sql_response', '')
        instance.dataframe = data.get('dataframe', [])

        return instance
    
    def __repr__(self):
        return (
            f"AIMessage(\n"
            f"  replied_user_message_id={self.replied_user_message_id!r},\n"
            f"  message_id={self.message_id!r},\n"
            f"  message={self.message!r},\n"
            f"  message_type={self.message_type!r},\n"
            f"  date_created={self.date_created.isoformat()!r},\n"
            f"  sql_response={self.sql_response!r},\n"
            f"  dataframe={self.dataframe if self.dataframe is None else 'DataFrame(...)'},\n"
            f")"
        )
        



class HumanMessage(Base_Message):
    def __init__(self, message="", message_type=MEMORY_TYPES["HUMAN"]) -> None:
        super().__init__(message, message_type)
        
    def to_dict(self) -> dict:
        return super().to_dict()

    def __repr__(self):
        return (
            f"HumanMessage(\n"
            f"  message_id={self.message_id!r},\n"
            f"  message={self.message!r},\n"
            f"  message_type={self.message_type!r},\n"
            f"  date_created={self.date_created.isoformat()!r}\n"
            f")"
        )
