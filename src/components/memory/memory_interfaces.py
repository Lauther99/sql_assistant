import sys
sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3")
from typing import Optional
import pandas as pd
from abc import ABC
import uuid
from datetime import datetime

class Base_Message(ABC):
    def __init__(self):
        self.message_id = uuid.uuid4()
        self.message = ""
        self.message_type = ""
        self.date_created = datetime.now()


class AIMessage(Base_Message):
    def __init__(self, user_message_id: uuid) -> None:
        super().__init__()
        self.dataframe: Optional[pd.DataFrame] = None
        self.replied_user_message_id = user_message_id

    def __repr__(self):
        return f"AIMessage(message={self.message!r}, dataframe={self.dataframe!r})"


class HumanMessage(Base_Message):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self):
        return f"HumanMessage(message={self.message!r})"
