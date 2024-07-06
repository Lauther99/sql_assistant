from src.components.memory.memory import Memory
from src.components.memory import MEMORY_TYPES


generate_request_template: str = """The following is a conversation between a human and you.
CONVERSATION:
{conversation}
END OF CONVERSATION

Your task is to look at the last human message, analyze it with all previous messages and briefly describe his intention and what the human wants to do or ask. If the last message refers to previous messages, add necessary information from previous messages in the final request of your response.

Do not include any explanations or apologies in your responses.
Do not add your own conclusions or clarifications.

Your answer might answer to: What is human requesting? or what is human doing?."""

generate_request_suffix = """Note: You may add sensitive information from  previous messages to the request if it is necessary to understand the human's intention or request.
Use the following key format to respond:
intention: The human is . . .

Begin!"""

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
    current_messages = memory.get_current_messages()
    conversation = ""
    for message in current_messages:
        m = message["content"]
        if message["type"] == MEMORY_TYPES["AI"]:
            conversation += f"AI Message: {m}\n"
        else:
            conversation += f"Human Message: {m}\n"

    prompt = generate_request_template.format(conversation=conversation)

    return prompt, generate_request_suffix


def get_simple_filter_prompt(user_request: str, examples: tuple):
    examples_text = ""

    for result in examples:
        examples_text += f"input: {result[1][1]}\n"
        examples_text += f"analysis: {result[0][1]}\n"
        examples_text += f"type: {result[2][1]}\n"

    prompt = simple_classifier_chain_template.format(examples=examples_text)

    suffix = simple_classifier_chain_suffix.format(user_request=user_request)

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
