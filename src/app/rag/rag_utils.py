
from src.components.models.models_interfaces import Base_LLM
from src.utils.utils import txt_2_Json


def base_llm_generation(model: Base_LLM, input, response_type):
    res = model.query_llm(input)
    try:
        response = txt_2_Json(str(res["text"]))
    except:
        response = None
    
    model.add_response(response_type, input, res["text"], response)
    return response
