from abc import ABC
from src.components.memory.memory import Memory
from src.components.collector.collector import AppDataCollector, LLMResponseCollector
from src.components.models.embeddings.embeddings import (
    HF_MultilingualE5_Embeddings,
    Openai_Embeddings,
)
from src.components.models.llms.llms import HF_Llama38b_LLM, Openai_LLM


class ChatConfig(ABC):
    def __init__(
        self,
        openai_llm: Openai_LLM,
        hf_llm: HF_Llama38b_LLM,
        mle5_embeddings: HF_MultilingualE5_Embeddings,
        openai_embeddings: Openai_Embeddings,
        collector: AppDataCollector,
        llm_collector: LLMResponseCollector,
        memory: Memory,
    ):
        self.memory: Memory = memory
        
        self.collector: AppDataCollector = collector
        self.llm_collector: LLMResponseCollector = llm_collector
        self.openai_llm: Openai_LLM = openai_llm
        self.hf_llm: HF_Llama38b_LLM = hf_llm
        self.mle5_embeddings: HF_MultilingualE5_Embeddings = mle5_embeddings
        self.openai_embeddings: Openai_Embeddings = openai_embeddings
        
        self.openai_llm.init_model()
        self.hf_llm.init_model()
        self.mle5_embeddings.init_model()
        self.openai_embeddings.init_model()
