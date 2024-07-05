import os
import pandas as pd
from typing import List, Dict

def read_tables_descriptions() -> Dict[str, List[str]]:
    """Obtiene data de las descripciones"""
    current_dir = os.path.dirname(__file__)
    excel_path = os.path.join(current_dir, "./data/sql_examples_data.xlsx")

    dataframe = pd.read_excel(io=excel_path, sheet_name="descriptions")
    dataframe_sin_nan = dataframe.dropna()

    data = []

    # Itera sobre cada fila del DataFrame
    for _, fila in dataframe_sin_nan.iterrows():
        datos_fila = {
            "table_name": fila["tables"],
            "descriptions": fila["descriptions"],
            "relations": fila["relations"],
            "aka_name": fila["aka_name"],
            # "keywords": str(fila["keywords"]).split(","),
            # "ddl": fila["ddls"],
        }
        data.append(datos_fila)

    return data

def read_tables_data():
    """Obtiene data de los ddls y documentacion"""
    current_dir = os.path.dirname(__file__)
    excel_path = os.path.join(current_dir, "./data/sql_examples_data.xlsx")

    ddl_df = pd.read_excel(io=excel_path, sheet_name="ddls").dropna()
    doc_df = pd.read_excel(
        io=excel_path, sheet_name="documentation", usecols=["table", "documentation"]
    ).dropna()

    return [ddl_df, doc_df]

def read_sql_examples():
    """Obtiene data de los ejemplos sql"""
    current_dir = os.path.dirname(__file__)
    excel_path = os.path.join(current_dir, "./data/sql_examples_data.xlsx")

    ex_df = pd.read_excel(io=excel_path, sheet_name="examples").dropna()
    return ex_df

def read_classify_dictionary():
    """Obtiene data para la clasificacion de requests"""
    current_dir = os.path.dirname(__file__)
    excel_path = os.path.join(current_dir, "./data/classifier_data.xlsx")

    dataframe = pd.read_excel(
        excel_path, usecols=["input", "analysis", "response"]
    ).dropna()
    return dataframe

def read_database_semantics(sheet_name: str  = "semantics_tables", cols: list[str] = None):
    """Obtiene data de semantics"""
    current_dir = os.path.dirname(__file__)
    excel_path = os.path.join(current_dir, "./data/semantics_data.xlsx")
    if cols is not None:
        ex_df = pd.read_excel(io=excel_path, sheet_name=sheet_name, usecols=cols)
    else: 
        ex_df = pd.read_excel(io=excel_path, sheet_name=sheet_name)
        
    return ex_df

# #TODO: ESTO DEBE IR EN EXPERIMENTS--------------------------------------------------------------
# def read_database_experiments(sheet_name: str  = "Hoja1", cols: list[str] = None):
#     """To read an excel that contains sql QA examples"""
#     current_dir = os.path.dirname(__file__)
#     excel_path = os.path.join(current_dir, "../../../assets/experiments.xlsx")
#     if cols is not None:
#         ex_df = pd.read_excel(io=excel_path, sheet_name=sheet_name, usecols=cols)
#     else: 
#         ex_df = pd.read_excel(io=excel_path, sheet_name=sheet_name)
        
#     return ex_df
# #TODO: ESTO DEBE IR EN EXPERIMENTS--------------------------------------------------------------

