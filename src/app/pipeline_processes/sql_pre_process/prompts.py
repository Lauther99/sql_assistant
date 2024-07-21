from src.components.memory import MEMORY_TYPES
from src.components.memory.memory import Memory
from src.components.models.models_interfaces import Base_Embeddings
from src.utils.reader_utils import read_tables_descriptions
from src.components.collector.collector import AppDataCollector


related_semantic_tables_template: str = """According to this tables descriptions:
{descriptions}
Use this relationships descriptions to find the most related tables to the user request:
{relations_descriptions_text}

Your task is to pick the most related tables according to the next user request:
user_request: {user_request}"""

related_semantic_tables_suffix: str = """Note: Yoy have to answer with database table name instead of the name of the table.

Use the following key format to respond:
tables: Comma separated list of selected tables.

Begin!"""

related_semantic_tables_template: str = """According to this tables descriptions:
{descriptions}
Use this relationships descriptions to find the most related tables to the user request:
{relations_descriptions_text}

Your task is to pick the most related tables according to the next user request:
user_request: '''{user_request}'''

Note: Yoy have to answer with database table name instead of the name of the table.

Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
Start Key format:
"tables" is the key and its content is: Comma separated list of selected tables.
End of Key format

Begin!"""

related_semantic_tables_suffix: str = """tables: """

multi_definition_question_template = """Your task is to generate a question about what should be given by user to make his request clear and reduce ambiguity, follow carefully the next steps:

First, look up the user request:
user_request: '''{user_request}'''

Second, read carefully the next analysis why is this request determinated as "unclear":
analysis: '''{analysis}'''

Third, evaluation. Before you generate a question, evaluate what is necessary to ask the user and what needs to be given by user to be a clear sentence based on the previous analysis.

Note: Don't forget to provide some options so the user can easily choose.

Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
Start Key format:
key: "question"
content: Generated question that will be asked to the user.
key: "analysis"
content: brief analysis in one line.
End of Key format

Begin!"""

multi_definition_question_suffix = "question:"

complement_request_template = """Your task is to improve the user request according to a selected option:

First, look up the sentence request (This is what you will modify):
sentence_request: '''{user_request}'''

Second, read carefully the next messages between AI and User:
ai_question: '''{ai_question}'''
user_response: '''{user_response}'''

Third, evaluation. Improve the previous request according to the user information given in his response.

Note: Try not to change all the sentence request, use user response information to complete the request.

Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
Start Key format:
key: "modified_sentence"
content: Modified sentence request, should start with: "The human is ...".
key: "analysis"
content: brief analysis in one line.
End of Key format

Begin!"""

complement_request_suffix = "modified_sentence:"

technical_terms_template = """I want to create a dictionary but I need your help to find all possible technical, ambiguous or unknowing terms in the sentence, here are some examples:
{terms_examples}
End of examples

Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
"terms" is the key and its content is: Comma separated list of terms . . .
End of Key format

Begin!
sentence: '''{user_request}'''"""

technical_terms_suffix = "terms:"

multi_definition_detector_prompt = """Your task is to classify the sentence into clear or unclear, follow carefully the next steps:

First, look up the technical terms found in the sentence:
Technical terms: '''{technical_terms}'''

Second, look up the next definitions and evaluate if there are multi definitions for each of technical terms shown before.
Definitions:
{definitions}
Third, evaluation. To classify pick every term and answer the question: what is human refering with this term?
If there is single definition for a term, then is clear, but if there are many definitions and with sentence context still making ambiguity, then is unclear. On the other hand, if there are many definitions and with sentence context human refers to specific definition, then is clear.

Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
Start Key format:
key: "class"
content: clear/unclear.
key: "analysis"
content: brief analysis.
End of Key format
Begin!
sentence: '''{sentence}'''"""

multi_definition_detector_suffix = "class:"

replace_terms_prompt = """The next is a request sentence:
sentence: '''{sentence}'''

Instructions:
Your task is to find the next list of technical terms in sentence and replace them if it is necessary with correct words for a better comprehension. 

Technical terms: '''{technical_terms}'''

Suggestions for replacing:
{replace_instructions}
Note: Pay attention to the recommendations. When you find a term that needs to be replaced use the suggested one. Do not add any other comments other than the modified sentence. Do not add your opinions.

Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
Start Key format:
key: "modified_sentence"
content: Modified sentence.
End of Key format

Begin!"""

replace_terms_suffix = "modified_sentence:"

enhanced_request_instruction: str = """I need your help with a very important task for my work. Follow carefully this instructions step by step.

First, read this current user intent:
The next is a conversation between Assistant and Human.
<user_intent>{user_intent}</user_intent> 

Second, the next are relevant slots from a conversation that are necessary to complete the previous user intent:
<Slots>{current_slots}</Slots>

Third, complementation. Add to the user intent the necessary slots to generate a complete and better user intent.

Finally, evaluation:
- The user intent should contain the relevant slots. A correct intent should answer the question: "What does the user want?". 
- Do not add your own ideas or clarifications or recommendations.

Output format response:
The output should be formatted with the key format below. Do not add anything beyond the key format.
Start Key format:
"response" is the key and its content: Detailed current user goal, it may starts with "The user is ...".
End of Key format

Begin!"""
enhanced_request_suffix = "response: "


def get_enhanced_request_prompt(collector: AppDataCollector):
    current_user_intent = collector.user_request
    current_slots = collector.current_conversation_data.current_slots


    instructions = enhanced_request_instruction.format(
        user_intent=current_user_intent,
        current_slots=current_slots
    )
    suffix = enhanced_request_suffix

    return instructions, suffix


def get_generate_semantic_tables_prompt(
    user_request,
    semantic_tables,
    semantic_relations_descriptions,
):
    data = read_tables_descriptions()
    relevant_descriptions = [
        (d["table_name"], d["descriptions"], d["aka_name"])
        for d in data
        if d["table_name"] in semantic_tables
    ]
    descriptions = (
        "\n---------------------------------------------------------------\n".join(
            [
                f"{doc[2]} table: {doc[1]}\nDatabase table name: {doc[0]}"
                for doc in relevant_descriptions
            ]
        )
    )
    relations_descriptions_text = "\n".join(
        [f" - {doc}" for doc in semantic_relations_descriptions]
    )
    prompt = related_semantic_tables_template.format(
        descriptions=descriptions,
        relations_descriptions_text=relations_descriptions_text,
        user_request=user_request,
    )

    return prompt, related_semantic_tables_suffix


def get_multi_definition_question_prompt(user_request, analysis):
    """Recomendable usar con el modelo de llama 3 por el suffix, para asi no tener inconvenientes con la respuesta."""
    return (
        multi_definition_question_template.format(
            user_request=user_request, analysis=analysis
        ),
        multi_definition_question_suffix,
    )


def get_complement_request_prompt(user_request, ai_question, user_response):
    """Recomendable usar con el modelo de llama 3 por el suffix, para asi no tener inconvenientes con la respuesta."""
    return (
        complement_request_template.format(
            user_request=user_request,
            ai_question=ai_question,
            user_response=user_response,
        ),
        complement_request_suffix,
    )


def get_technical_terms_prompt(user_request, terms_examples_results):
    """Recomendable usar con el modelo de llama 3 por el suffix, para asi no tener inconvenientes con la respuesta."""
    examples = []
    for item in terms_examples_results:
        example = f"""sentence: {item[1][1]}\nterms: {item[0][1]}"""
        examples.append(example)

    terms_examples_text = f"""\n{"-"*50}\n""".join(examples)

    return (
        technical_terms_template.format(
            user_request=user_request, terms_examples=terms_examples_text
        ),
        technical_terms_suffix,
    )


def get_multi_definition_detector_prompt(user_request, terms_dictionary):
    """Recomendable usar con el modelo de llama 3 por el suffix, para asi no tener inconvenientes con la respuesta."""

    definitions = ""
    technical_terms_arr = list()
    for _, item in enumerate(terms_dictionary):
        technical_terms_arr.append(item["original_term"])
        definitions += f"""For term '{item["original_term"]}'\n"""
        for _, inner_definition in enumerate(item["definitions"]):
            definitions += f"""- {inner_definition["definition"]}\n"""
        definitions += "\n"

    technical_terms = ", ".join(technical_terms_arr)

    instruction = multi_definition_detector_prompt.format(
        technical_terms=technical_terms, definitions=definitions, sentence=user_request
    )

    return instruction, multi_definition_detector_suffix


def get_modified_request_prompt(user_request, terms_dictionary):
    replace_instructions = ""
    technical_terms_arr = list()
    for _, item in enumerate(terms_dictionary):
        if len(item["definitions"]) > 0:
            technical_terms_arr.append(item["original_term"])
            replace_instructions += f"""For term '{item["original_term"]}'\n"""
            for _, inner_definition in enumerate(item["definitions"]):
                replace_instructions += (
                    f"""- {inner_definition["replace_instruction"]}\n"""
                )
            replace_instructions += "\n"

    technical_terms = ", ".join(technical_terms_arr)

    instruction = replace_terms_prompt.format(
        technical_terms=technical_terms,
        replace_instructions=replace_instructions,
        sentence=user_request,
    )

    return instruction, replace_terms_suffix
