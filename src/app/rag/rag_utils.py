
from src.components.collector.collector import LLMResponseCollector
from src.components.models.models_interfaces import Base_LLM
from src.utils.utils import txt_2_Json


def base_llm_generation(llm: Base_LLM, llm_collector: LLMResponseCollector, prompt, response_type):
    res = llm.query_llm(prompt)
    print(response_type)
    print(res, "\n\n")
    try:
        response = txt_2_Json(str(res["text"]))
    except:
        response = None
    
    llm_collector.add_response(response_type, prompt, str( res["text"]).strip(), response)
    
    return response
