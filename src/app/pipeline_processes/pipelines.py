from src.components.collector.collector import AppDataCollector, LLMResponseCollector
from src.components.memory.memory import Memory
from src.components.models.models_interfaces import Base_Embeddings, Base_LLM
from src.app.pipeline_processes.query_post_process.manager import query_post_process
from src.app.pipeline_processes.query_pre_process.manager import (
    query_pre_process,
    simple_request_process,
)
from src.app.pipeline_processes.sql_generation_process.manager import (
    complex_request_sql_generation,
)
from src.app.pipeline_processes.sql_post_process.manager import (
    complex_request_pre_query_generation,
    complex_request_sql_summary_response,
    complex_request_sql_verification,
)
from src.app.pipeline_processes.sql_pre_process.manager import (
    complex_request_process_modification,
    complex_request_process_semantics,
)
from src.utils.sql_utils import run_sql
import traceback


def _pre_process_pipeline(
    llm: Base_LLM,
    memory: Memory,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
):
    try:
        query_pre_process(
            llm=llm,
            memory=memory,
            collector=collector,
            llm_collector=llm_collector,
        )
    except Exception as e:
        print(f"Error en: query_pre_process:\n {e}")
        traceback.print_exc()


def _simple_request_pipeline(
    llm: Base_LLM, collector: AppDataCollector, llm_collector: LLMResponseCollector
):
    try:
        simple_request_process(
            llm=llm,
            collector=collector,
            llm_collector=llm_collector,
        )
    except Exception as e:
        print(f"Error en: simple_request_process:\n {e}")
        traceback.print_exc()


def _complex_request_pipeline(
    llama3_llm: Base_LLM,
    openai_llm: Base_LLM,
    mle5_embeddings: Base_Embeddings,
    openai_embeddings: Base_Embeddings,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
    memory: Memory,
):
    try:
        print("complex_request_process_modification")
        complex_request_process_modification(
            llm=llama3_llm,
            embeddings=mle5_embeddings,
            collector=collector,
            llm_collector=llm_collector,
        )
    except Exception as e:
        print(f"Error en: complex_request_process_modification:\n {e}")
        traceback.print_exc()

    try:
        print("complex_request_process_semantics")
        complex_request_process_semantics(
            llm=llama3_llm,
            embeddings=openai_embeddings,
            collector=collector,
            llm_collector=llm_collector,
        )
    except Exception as e:
        print(f"Error en: complex_request_process_semantics:\n {e}")
        traceback.print_exc()

    try:
        print("complex_request_sql_generation")
        complex_request_sql_generation(
            llm=openai_llm, collector=collector, llm_collector=llm_collector
        )
    except Exception as e:
        print(f"Error en: complex_request_sql_generation:\n {e}")
        traceback.print_exc()

    # !Este es el prequery, hay que mejorar el prompt (puede funcionar con un fine tune)
    # try:
    #     print("complex_request_sql_verification")
    #     complex_request_sql_verification(
    #         llm=llama3_llm, collector=collector, llm_collector=llm_collector
    #     )
    # except Exception as e:
    #     print(f"Error en: complex_request_sql_verification:\n {e}")
    #     traceback.print_exc()

    # is_prequery = collector.assistant_sql_code_class.strip() == "incomplete"
    # if is_prequery:
    #     try:
    #         print("complex_request_pre_query_generation")
    #         complex_request_pre_query_generation(
    #             llm=openai_llm, collector=collector, llm_collector=llm_collector
    #         )
    #     except Exception as e:
    #         print(f"Error en: complex_request_pre_query_generation:\n {e}")
    #         traceback.print_exc()


def _post_sql_generation_pipeline(
    llama3_llm: Base_LLM,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
):
    df = [{'Response': "Empty"}]
    try:
        print("Ejecutando codigo SQL ...")
        df = run_sql(collector.sql_code)
    except Exception as e:
        print(f"Error al ejecutar collector.sql_code \n{e}")
        traceback.print_exc()

    try:
        complex_request_sql_summary_response(
            llm=llama3_llm,
            collector=collector,
            llm_collector=llm_collector,
            dataframe=df,
        )
    except Exception as e:
        print(f"Error al ejecutar complex_request_sql_summary_response\n{e}")
        traceback.print_exc()
    # is_prequery = (
    #     collector.assistant_sql_code_class.strip() == "incomplete"
    #     if collector.assistant_sql_code_class
    #     else False
    # )

    # if is_prequery:
    #     try:
    #         print("Ejecutando codigo SQL de prequery ...")
    #         df = run_sql(collector.sql_pre_query)
    #     except Exception as e:
    #         print(f"Error al ejecutar collector.sql_pre_query \n{e}")
    #         traceback.print_exc()
    #     try:
    #         complex_request_sql_summary_response(
    #             llm=llama3_llm,
    #             collector=collector,
    #             llm_collector=llm_collector,
    #             dataframe=df,
    #         )
    #     except Exception as e:
    #         print(f"Error al ejecutar complex_request_sql_summary_response\n{e}")
    #         traceback.print_exc()
    # else:
    #     try:
    #         print("Ejecutando codigo SQL ...")
    #         df = run_sql(collector.sql_code)
    #     except Exception as e:
    #         print(f"Error al ejecutar collector.sql_code \n{e}")
    #         traceback.print_exc()

    #     try:
    #         complex_request_sql_summary_response(
    #             llm=llama3_llm,
    #             collector=collector,
    #             llm_collector=llm_collector,
    #             dataframe=df,
    #         )
    #     except Exception as e:
    #         print(f"Error al ejecutar complex_request_sql_summary_response\n{e}")
    #         traceback.print_exc()


def _post_process_pipeline(
    openai_llm: Base_LLM,
    collector: AppDataCollector,
    llm_collector: LLMResponseCollector,
):
    try:
        query_post_process(
            llm=openai_llm,
            collector=collector,
            llm_collector=llm_collector,
        )
    except Exception as e:
        print(f"Error al ejecutar query_post_process\n{e}")
        traceback.print_exc()
