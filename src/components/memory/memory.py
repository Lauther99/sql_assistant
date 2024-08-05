import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3")
from src.db.mongo.interfaces import ChatDocument
from src.components.memory.memory_interfaces import AIMessage, HumanMessage
from src.components.memory import MEMORY_TYPES
from typing import Any, Hashable, List, Optional, Union
import pandas as pd

summary_instruction_with_dictionary: str = """I need your help with a very important task for my work. Follow carefully this instructions step by step.

First, read this current summary and prevalent slots of a conversation between a virtual assistant and a user:
<Summary>{current_summary}</Summary>
<Slots>{current_slots}</Slots>

Second, read this new lines to the conversation:
{chat_history}

Third, read this term definitions from dictionary:
{dictionary}

Fourth, summarization. Summarize step by step the new lines and add this new summary to the previous summary.

Fifth, user intent. Find the exactly current user intention at this particular point of the conversation. Be specific by mentioning the slots and any other data mentioned in conversation.

Sixth, slots. Find the prevalent slots to the conversation.

Finally, evaluation:
- The summary should describe step by step the conversation. A correct summary should contains the slots mentioned in conversation and answer the question: "What does user and assistant talking about?".
- The user intent should contain the relevant slots he mention along the conversation. A correct intent should answer the question: "What does the user want at this point in the conversation?".

Note that it is possible that not all conversation history is relevant and you need to summarise based on what is relevant. If the user does not have a goal at this point the intent just aim on what is the user doing: “The user is ...”.

Output format response:
The output should be formatted with the key format below. Do not add any comment before or after the key format. Just this format please.
Start Key format:
"summary" is the key and its content: Detailed summary of the current conversation in one line, do not make break lines.
"user_intent" is the key and its content: Detailed current user goal at this point of the conversation.
"slots" is the key and its content: Comma separated dictionary with slots and its values identified at this point of the conversation in one line, do not make break lines.
End of Key format

Begin!"""

summary_instruction_no_dictionary: str = """I need your help with a very important task for my work. Follow carefully this instructions step by step.

First, read this current summary and prevalent slots of a conversation between a virtual assistant and a user:
<Summary>{current_summary}</Summary>
<Slots>{current_slots}</Slots>

Second, read this new lines to the conversation:
{chat_history}

Third, summarization. Summarize step by step the new lines and add this new summary to the previous summary.

Fourth, user intent. Find the exactly current user intention at this particular point of the conversation.

Fifth, slots. Find the prevalent slots that are necessary to the current intention.

Finally, evaluation:
- The summary should describe step by step the conversation. A correct summary should contains the slots mentioned in conversation and answer the question: "What does user and assistant talking about?".
- The user intent should contain the relevant slots he mention along the conversation. A correct intent should answer the question: "What does the user want at this point in the conversation?". 

Note that it is possible that not all conversation history is relevant and you need to summarise based on what is relevant. If the user does not have a goal at this point the intent just aim on what is the user doing: “The user is ...”.

Output format response:
The output should be formatted with the key format below. Do not add any comment before or after the key format. Just this format please.
Start Key format:
"summary" is the key and its content: Detailed summary of the current conversation in one line, do not make break lines.
"user_intent" is the key and its content: Detailed current user goal at this point of the conversation.
"slots" is the key and its content: Comma separated dictionary with slots and its values identified at this point of the conversation in one line, do not make break lines.
End of Key format

Begin!"""

summary_instruction_suffix = """summary: """


class Memory:
    
    def __init__(self, chat_document: ChatDocument) -> None:
        self.chat_memory = chat_document.messages
    
    # Chat management
    def add_user_message(
        self, message: str
    ):
        # ToDo: Llamada a la base de datos para agregar el mensaje al chat
        human_message = HumanMessage(message, MEMORY_TYPES["HUMAN"])
        self.chat_memory.append(human_message)
        return human_message

    def add_ai_message(
        self,
        message: str,
        replied_user_message_id: str,
        df: list[dict[Hashable, Any]] = None,
        sql_response: str | None = ""
    ):
        # ToDo: Llamada a la base de datos para agregar el mensaje al chat
        ai_message = AIMessage(replied_user_message_id, message, MEMORY_TYPES["AI"], df, sql_response)
        self.chat_memory.append(ai_message)
        return ai_message

    def get_new_summary_instruction(
        self,
        user_message: HumanMessage,
        ai_message: AIMessage = None,
        current_summary: str = None,
        current_slots: str = None,
        dictionary = None,
    ):
        current_summary = "The next is a conversation between Assistant and Human." if current_summary is None else current_summary
        new_lines = ""
        if ai_message is None:
            new_lines = f"""<User>{user_message.message}</User>"""
        else:
            if ai_message.dataframe is not None:
                new_lines = f"""<Assistant>\n{ai_message.message}\nHere is a dataframe: \n{pd.DataFrame(ai_message.dataframe).head(10)}\n</Assistant>\n<User>{user_message.message}</User>"""
            else:
                new_lines = f"""<Assistant>{ai_message.message}</Assistant>\n<User>{user_message.message}</User>"""

        if dictionary is not None:
            definitions = []
            for item in dictionary:
                for inner_item in item["definitions"]:
                    definitions.append(str(inner_item["definition"]).strip())
            terms = "    - "
            content = "\n    - ".join(definitions)
            terms += f"{content}\n"
                
            instruction = summary_instruction_with_dictionary.format(
                current_summary=current_summary,
                current_slots=current_slots,
                chat_history=new_lines,
                dictionary=terms
            )
        else:
            instruction = summary_instruction_no_dictionary.format(
                current_summary=current_summary,
                current_slots=current_slots,
                chat_history=new_lines,
            )
        
        suffix = summary_instruction_suffix
        return instruction, suffix

    def list_chat_messages(self):
        new_lines: str = ""
        try:
            if len(self.chat_memory) > 0:
                for message in self.chat_memory:
                    if message.message_type == MEMORY_TYPES["HUMAN"]:
                        message_content = message.message
                        new_lines += (
                            f"\n<Human>: {message.date_created}\n{message_content}"
                        )
                    elif message.message_type == MEMORY_TYPES["AI"]:
                        if message.dataframe is not None:
                            new_lines += f"\n<Assistant>: {message.date_created}\n{message.message}\nHere is a dataframe from SQL: \n{pd.DataFrame(message.dataframe).head(10).to_markdown()}"
                        else:
                            new_lines += f"\n<Assistant>: {message.date_created}\n{message.message}"
                    else:
                        message_type = message.message_type
                        raise ValueError(f"Not support message type: {message_type}")
            else:
                raise ValueError("Not messages found in the current conversation")
            return new_lines
        except Exception as e:
            print(e)
            return new_lines

        
