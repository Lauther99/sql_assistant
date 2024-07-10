import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3")
from src.components.collector.collector import LLMResponseCollector
from src.app.rag.rag_utils import base_llm_generation
from src.settings.settings import Settings
from src.components.models.models_interfaces import Base_LLM
from src.components.memory import MEMORY_TYPES
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Optional
import pandas as pd
from abc import ABC, abstractmethod

memory_prompt_instruction: str = """Follow carefully the next steps:

First, look up at the next current lines in a conversation between Human and Assistant
Current lines:
{chat_history}

Second, look up the current conversation summary
Current summary:
'''{current_summary}'''

Third, progressively summarize the new lines provided and merge with previous summary returning a new summary.

Fourth, evaluation. Evaluate if your new summary answer the question: What is the conversation about?
 
Note:
 - Be detailed with important and sensitive information. 
 - You may add sensitive information to your response, like names or technical terms that are mentioned in conversation.
 - Do not hallucinate or try to predict the conversation, work exclusively with the new lines.
 - Do not include any explanations or apologies in your response.
 - Do not add your own conclusions or clarifications.
 - Use third grammatical person in your summary. 
 - DO NOT give an empty response.

Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
Start Key format:
"new_summary" is the key and its content is: Summary of the conversation.
End of Key format

Begin!"""

memory_prompt_suffix = """new_summary: """


class Base_Message(ABC):
    def __init__(self):
        self.message = ""
        self.message_type = ""


class AIMessage(Base_Message):
    def __init__(self) -> None:
        super().__init__()
        self.dataframe: Optional[pd.DataFrame] = None

    def __repr__(self):
        return f"AIMessage(message={self.message!r}, dataframe={self.dataframe!r})"


class HumanMessage(Base_Message):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self):
        return f"HumanMessage(message={self.message!r})"


class Memory:
    def __init__(self) -> None:
        self.chat_memory: list[Base_Message] = []

    # Chat management
    def add_user_message(self, message: str):
        human_message = HumanMessage()

        if (
            len(self.chat_memory) == 0
            or self.chat_memory[-1].message_type == MEMORY_TYPES["AI"]
        ):
            human_message.message = message
            human_message.message_type = MEMORY_TYPES["HUMAN"]
            self.chat_memory.append(human_message)
            return
        raise ValueError("Cannot add Human message")

    def add_ai_message(self, message: str, df: Optional[pd.DataFrame] = None):
        ai_message = AIMessage()
        if (
            len(self.chat_memory) > 0
            and self.chat_memory[-1].message_type == MEMORY_TYPES["HUMAN"]
        ):
            ai_message.message_type = MEMORY_TYPES["AI"]
            ai_message.message = message
            ai_message.dataframe = df
            self.chat_memory.append(ai_message)
            return
        raise ValueError("Can not add AI message")

    def get_summary_prompt_template(
        self,
        existing_summary: str = "The next is a conversation between Assistant and Human",
    ):
        new_lines: str = self.get_chat_history_lines(self.chat_memory)
        instruction = memory_prompt_instruction.format(
            current_summary=existing_summary, chat_history=new_lines
        )
        suffix = memory_prompt_suffix

        return instruction, suffix

    @staticmethod
    def get_chat_history_lines(chat_memory: list[Base_Message]):
        new_lines: str = ""
        if len(chat_memory) > 0:
            for message in chat_memory:
                if message.message_type == MEMORY_TYPES["HUMAN"]:
                    message_content = message.message
                    new_lines += f"\nHuman line: {message_content}"
                elif message.message_type == MEMORY_TYPES["AI"]:
                    new_lines += f"\nAssistant line: {message.message}"
                    if message.dataframe is not None:
                        new_lines += f"\nAssistant DataFrame: \n{message.dataframe.head(10).to_markdown()}"
                else:
                    message_type = message.message_type
                    raise ValueError(f"Not support message type: {message_type}")
        else:
            raise ValueError("Not messages found in the current conversation")

        return new_lines

    def clear_memory(self):
        self.chat_memory = []
