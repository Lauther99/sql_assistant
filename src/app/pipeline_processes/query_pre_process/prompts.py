from src.components.memory.memory import Memory
from src.components.memory import MEMORY_TYPES

# Estos son generate_request de openai
generate_request_template: str = """The following is a conversation between a human and you.
CONVERSATION:
{chat_history}
END OF CONVERSATION

Your task is to look at the last human message, analyze it with all previous messages and briefly describe his intention and what the human wants to do or ask. If the last message refers to previous messages, add necessary information from previous messages in the final request of your response.

Do not include any explanations or apologies in your responses.
Do not add your own conclusions or clarifications.

Your answer might answer to: What is human requesting? or what is human doing?."""

generate_request_suffix = """Note: You may add sensitive information from  previous messages to the request if it is necessary to understand the human's intention or request.
Use the following key format to respond:
intention: The human is . . .

Begin!"""
# Estos son generate_request de openai

# Estos son generate_request de llama
generate_request_template: str = """I need your help, I'm trying to generate a summarize request from a user in a conversation.

Following the next conversation
{chat_history}
END OF CONVERSATION

Your task is to analyze the messages and briefly describe in one line the human intention and what he is looking for or doing.

Here are some advices for a better response:
 - Pay attention if the last message refers to previous ones to add necessary information located in previous messages.
 - You may add sensitive information to your response, like names or technical terms that are mentioned in conversation.
 - Do not include any explanations or apologies in your response.
 - Do not add your own conclusions or clarifications.
 - Do not add words nor nouns nor adjectives to complement the response if these are not mentioned in the conversation.

Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
Start Key format:
"intention" is the key and its content is: Detailed user request. It may start with The human is . . .
End of Key format

Begin!"""

generate_request_suffix = """intention: """
# Estos son generate_request de llama

# Estos son simple_classifier de openai
simple_classifier_chain_template: str = """Your task is to classify the input_request into one of the following categories: simple/complex
simple: When the input_request is simple to answer with greetings or any other input_request intent that is NOT related to measurement systems database.
complex: When the intent of input_request is related to get from information from database. And this information is in a database that you have NO access.

The next is information you have to know before classify, is that there is a measurement system database, but you do not have access to this database.
The only thing you know is that, if you had access you could answer questions related to measurement systems, but you don't.
So if input_request is related to get information from this database it would be complex.

Follow this examples:
{examples}
End of examples"""

simple_classifier_chain_suffix = """Use the following format to respond:
analysis: Your analysis for the input_request.
type: complex/simple

Begin!
input_request: {user_request}"""
# Estos son simple_classifier de openai

# Estos son simple_classifier de llama
simple_classifier_chain_template: str = """Your task is to classify the input_request into one of the following categories: simple/complex
simple: When the input_request is simple to answer with greetings or any other input_request intent that is NOT related to measurement systems database.
complex: When the intent of input_request is related to get from information from database. And this information is in a database that you have NO access.

The next is information you have to know before classify, is that there is a measurement system database, but you do not have access to this database.
The only thing you know is that, if you had access you could answer questions related to measurement systems, but you don't.
In that way, if input_request is related to get information from this database it would be complex.

Use this examples to guide your answer:
{examples}
End of examples

Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
Start Key format:
key: "type"
content: complex/simple.
key: "analysis"
content: analysis for your classification.
End of Key format

Begin!
input_request: '''{user_request}''' """

simple_classifier_chain_suffix = """type: """
# Estos son simple_classifier de llama

greeting_chain_template: str = """This are your capabilities:
- You can greet people.
- If you are asked for a task regarding measurement systems, respond that you can help obtaining the following information of temperature, pressure, viscosity, among other parameters of the existing measurement systems in the database.
- If the user does not know what to ask you, then you can respond that you can help you obtaining the following information "List of measurement systems", "List of meters for a specific measurement system", "Average temperature for specific measuring system"

Your task is to continue the following conversation:
{conversation}"""

greeting_chain_suffix = """Note: Do not create new user messages, only respond as M-Assistant. Do not respond with any additional explanation beyond the conversation. Answer once.

Use the following format to respond:
message: Your response to attend the user.

Begin!"""


def get_generate_request_prompt(memory: Memory):
    chat_history = memory.get_chat_history_lines(memory.chat_memory)
    prompt = generate_request_template.format(chat_history=chat_history)
    return prompt, generate_request_suffix


def get_simple_filter_prompt(user_request: str, examples: tuple):
    examples_text = ""

    for result in examples:
        examples_text += f"input: {result[1][1]}\n"
        examples_text += f"analysis: {result[0][1]}\n"
        examples_text += f"""type: {result[2][1]}\n{"-"*30}\n"""

    prompt = simple_classifier_chain_template.format(examples=examples_text, user_request=user_request)

    suffix = simple_classifier_chain_suffix

    return prompt, suffix


def get_greeting_response_prompt(memory: Memory):
    current_messages = memory.get_current_messages()
    conversation = ""
    for message in current_messages:
        m = message["content"]
        if message["type"] == MEMORY_TYPES["AI"]:
            conversation += f"AI Message: {m}\n"
        else:
            conversation += f"Human Message: {m}\n"

    prompt = greeting_chain_template.format(conversation=conversation)

    return prompt, greeting_chain_suffix
