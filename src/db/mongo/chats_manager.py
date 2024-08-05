from src.settings.settings import Settings
from src.db.mongo.interfaces import ChatDocument
import uuid
import traceback
import datetime

def create_new_chat(user_id: uuid.UUID):
    client, collection = Settings.MongoDB.get_chats_collection()

    new_chat = ChatDocument()
    new_chat.user_id = user_id
    new_chat.conversation_id = uuid.uuid4()
    new_chat.last_interaction = datetime.datetime.now()

    new_chat_document = new_chat.chat_document_to_dict()
    try:
        collection.insert_one(new_chat_document).inserted_id
        print("Nuevo chat creado!")
        return new_chat.conversation_id
    except Exception as e:
        print(f"Error durante la conexion: {e}")
        traceback.print_exc()
    finally:
        client.close()
        print("Conexion finalizada")


def find_chat_by_id(chat_id: uuid.UUID):
    client, collection = Settings.MongoDB.get_chats_collection()

    try:
        chat = collection.find_one({"conversation_id": str(chat_id)})
        if chat is not None:
            print("Chat has been found")
            chat_document = ChatDocument.parse_dict_to_document(chat)
            return chat_document
        print("No se encontró ningún chat")

    except Exception as e:
        print(f"Error durante la conexion: {e}")
        traceback.print_exc()
    finally:
        client.close()
        print("Conexion finalizada")


def save_to_chat(new_data: ChatDocument):
    client, collection = Settings.MongoDB.get_chats_collection()

    conversation_id = new_data.conversation_id

    chat_dict = new_data.chat_document_to_dict()
    del chat_dict["user_id"]
    del chat_dict["conversation_id"]

    try:
        print("Guardando chat ...")
        collection.update_one(
            {"conversation_id": str(conversation_id)}, {"$set": chat_dict}, upsert=True
        )
        print("Save to chat successfully")

    except Exception as e:
        print(f"Error durante la conexion: {e}")
        traceback.print_exc()
    finally:
        client.close()
        print("Conexion finalizada")
