from src.db.handlers.handlers import query_by_texts
from src.settings.settings import Settings


def retrieve_sql_examples(
    user_request: str,
):
    collection = Settings.Chroma.get_sql_examples_collection()
    results = query_by_texts(collection, [user_request], 4, score_threshold=0.3)
    return results

