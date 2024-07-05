from src.db.handlers.handlers import query_by_texts
from src.settings.settings import Settings



def retrieve_classify_examples(user_request):
    collection = Settings.Chroma.get_classify_col()
    results = query_by_texts(
        collection=collection, texts=[user_request], n=7, score_threshold=0.5
    )
    return results