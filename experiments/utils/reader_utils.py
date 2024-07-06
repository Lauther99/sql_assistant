import os
import pandas as pd


def read_database_experiments(sheet_name: str  = "", cols: list[str] = None):
    """To read an excel that contains sql QA examples"""
    current_dir = os.path.dirname(__file__)
    excel_path = os.path.join(current_dir, "./data/experiments_data.xlsx")
    if cols is not None:
        ex_df = pd.read_excel(io=excel_path, sheet_name=sheet_name, usecols=cols)
    else: 
        ex_df = pd.read_excel(io=excel_path, sheet_name=sheet_name)
        
    return ex_df

