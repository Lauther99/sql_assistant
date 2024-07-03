import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3")
from src.settings.settings import Settings
from src.components.models.models import Langchain_Model
from src.components.memory import MEMORY_TYPES
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Optional
import pandas as pd

memory_template: str = """Your task is to rogressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary. 
Be detailed with important information. 
Do not hallucinate or try to predict the conversation, work exclusively with the new lines.
If summary and new lines are empty then return an empty summary.
The new summary will have to answer the question: "What is the conversation about?"

Begin!
{current_summary}

New lines of conversation:
{new_lines}

New summary:"""


class Memory:
    def __init__(self, model: Langchain_Model) -> None:
        self.chat_memory = []
        self.model = model

    def init_model(self):
        print("Iniciando el modelo para la memoria")
        self.model.init_model()

    def get_current_messages(self):
        return self.chat_memory

    # Chat management
    def add_user_message(self, message: str):
        if (
            len(self.chat_memory) == 0
            or self.chat_memory[-1]["type"] == MEMORY_TYPES["AI"]
        ):
            self.chat_memory.append(
                {
                    "type": MEMORY_TYPES["HUMAN"],
                    "content": message,
                }
            )
            return
        raise ValueError("Can not Human message")

    def add_ai_message(self, message: str, df: Optional[pd.DataFrame]=None):
        if (
            len(self.chat_memory) > 0
            and self.chat_memory[-1]["type"] == MEMORY_TYPES["HUMAN"]
        ):
            self.chat_memory.append(
                {"type": MEMORY_TYPES["AI"], "content": message, "df": df}
            )
            return
        raise ValueError("Can not add AI message")

    def predict_conversation_summary(self, existing_summary: str = ""):
        new_lines: str = ""
        prompt = memory_template
        existing_summary = f"Current summary:\n {existing_summary}" if len(existing_summary) >=1 else ""

        if len(self.chat_memory) > 0:
            for message in self.chat_memory:
                if message["type"] == MEMORY_TYPES["HUMAN"]:
                    message_content = message["content"]
                    new_lines += f"Human: {message_content}"
                elif message["type"] == MEMORY_TYPES["AI"]:
                    message_content = message["content"]
                    new_lines += f"AI: {message_content}"
                else:
                    message_type = message["type"]
                    raise ValueError(f"Not support message type: {message_type}")
        else:
            raise ValueError("Not messages found in the current conversation")

        prompt = prompt.format(current_summary=existing_summary, new_lines=new_lines)

        res = self.model.query_llm(prompt)

        return res["text"]

    def clear_memory(self):
        self.chat_memory = []

    def get_last_user_message(self):
        last_message_1 = self.chat_memory[-1]
        last_message_2 = self.chat_memory[-2]

        if last_message_1["type"] == MEMORY_TYPES["HUMAN"]:
            return last_message_1
        elif last_message_2["type"] == MEMORY_TYPES["HUMAN"]:
            return last_message_2
        else:
            return None
