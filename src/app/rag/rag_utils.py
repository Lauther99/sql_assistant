
from src.components.models.models_interfaces import Base_LLM
from src.utils.utils import txt_2_Json


def base_llm_generation(llm: Base_LLM, prompt, response_type):
    res = llm.query_llm(prompt)
    try:
        response = txt_2_Json(str(res["text"]))
    except:
        response = None
    
    llm.add_response(response_type, prompt, res["text"], response)
    return response
