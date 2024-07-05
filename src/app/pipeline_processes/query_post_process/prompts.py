translator_template = """Your task is to translate Text_2, follow this steps:
        
First, detect language in Text_1:
Text_1: '''{user_input}'''

Second, detect language in Text_2:
Text_2: '''{actual_answer}'''

Third, translate Text_2 into Text_1's language.

Use the following format to respond:
detected_language: english/spanish/portuguesse, etc
response: Text_2 translated into the detected language.
"""