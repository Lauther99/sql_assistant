from src.components.memory.memory import Memory
from src.components.memory import MEMORY_TYPES
from src.components.memory.memory_interfaces import HumanMessage, AIMessage


# Estos son simple_classifier de llama
simple_classifier_template: str = """Your task is to classify the input_request into one of the following categories: simple/complex
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

simple_classifier_suffix = """type: """
# Estos son simple_classifier de llama

greeting_template: str = """This are your capabilities:
- You can greet people.
- If you are asked for a task regarding measurement systems, respond that you can help obtaining the following information of temperature, pressure, viscosity, among other parameters of the existing measurement systems in the database.
- If the user does not know what to ask you, then you can respond that you can help you obtaining the following information "List of measurement systems", "List of meters for a specific measurement system", "Average temperature for specific measuring system"

Your task is to continue the following conversation:
{conversation}

Note: Do not create new user messages. Do not respond with any additional explanation beyond the conversation. Answer once.

Use the following format to respond:
message: Your response to attend the user.

Begin!"""

greeting_suffix = """message:"""


def get_simple_filter_prompt(user_request: str, examples: tuple):
    examples_text = ""

    for result in examples:
        examples_text += f"input: {result[1][1]}\n"
        examples_text += f"analysis: {result[0][1]}\n"
        examples_text += f"""type: {result[2][1]}\n{"-"*30}\n"""

    prompt = simple_classifier_template.format(examples=examples_text, user_request=user_request)

    suffix = simple_classifier_suffix

    return prompt, suffix


def get_greeting_response_prompt(last_user_message: HumanMessage, last_ai_message: AIMessage | None):
    last_ai_message = "" if last_ai_message is None else f"""<Assistant>{last_ai_message.message}</Assistant>"""
    chat_lines = f"""{last_ai_message}\n<User>{last_user_message.message}</User>"""
    
    prompt = greeting_template.format(conversation=chat_lines)

    return prompt, greeting_suffix
