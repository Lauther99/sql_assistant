translator_template = """Your task is to translate Text_2, follow this steps:
        
First, detect language in Text_1:
Text_1: '''{user_input}'''

Second, look up Text_2:
Text_2: '''{actual_answer}'''

Third, carefully translate Text_2 into Text_1's language."""

translator_template_suffix="""Use the following format to respond:
detected_language: detected language in Text_1.
response: Text_2 translated into the detected language."""

def get_translator_prompt(user_input, actual_answer):
    instruction = translator_template.format(user_input=user_input, actual_answer=actual_answer)
    
    return instruction, translator_template_suffix