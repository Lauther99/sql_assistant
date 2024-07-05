import pandas as pd
from src.settings.settings import Settings

def run_sql(sql: str) -> pd.DataFrame:
    """To query database with SQL code generated"""
    try:
        conn = Settings.Sql.get_pyodbc_connection_string()
        df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        print(f"Error al ejecutar la consulta SQL: {e}")
        return e.args