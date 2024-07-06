import nltk

nltk.data.path.append(
    "C:\\Users\\lauth\\OneDrive\\Desktop\\sql_assistant_v3\\src\\assets\\nltk_data"
)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from openai import OpenAI

import re
import json

from numpy import dot
from numpy.linalg import norm
from autocorrect import Speller


eng_stopwords = set(stopwords.words("english"))
spell = Speller(lang="en")


def txt_2_Json(txt: str, read_text: bool = False) -> dict[str, any]:
    """Convierte un texto con keys en un JSON"""
    if read_text:
        print("=" * 30, "TXT 2 JSON", "=" * 30)
        print(txt)
    txt = txt.replace("}", "").replace("{", "")
    lineas = [linea.strip() for linea in txt.strip().split("\n") if linea.strip()]
    pares = [linea.split(": ", 1) for linea in lineas]
    datos = {clave.strip().lower(): valor.rstrip(",") for clave, valor in pares}
    res = json.dumps(datos, indent=4)
    res = json.loads(res)
    return res


def clean_sentence(sentence: str) -> str:
    """Limpia una oracion"""
    lemmatizer = WordNetLemmatizer()
    no_symbols = clean_symbols(sentence)

    words = [
        lemmatizer.lemmatize(lemmatizer.lemmatize(word, "v"), "n")
        for word in no_symbols.split()
        if word not in eng_stopwords
    ]
    bigwords = " ".join(words)

    return bigwords


def clean_technical_term(sentence: str) -> str:
    """Limpia los terminos sin simbolos"""
    no_symbols = re.sub("[^a-zA-Z' ]", " ", sentence).lower().strip()
    cleaned_sentence = re.sub(" +", " ", no_symbols)
    return cleaned_sentence


def clean_symbols(sentence: str) -> str:
    """Limpia los terminos numeros"""
    no_symbols = re.sub("[^a-zA-Z0-9' ]", "", sentence).lower().strip()
    return no_symbols


def string_2_array(t: str) -> list[str]:
    """Convierte una cadena en un arreglo"""
    r = t.replace("[", "").replace("]", "").split(",")
    r = [elemento.strip() for elemento in r]
    return r


def string_2_bool(cadena: str):
    """Convierte una cadena en booleano"""
    palabras = cadena.lower().split()
    return "true" in palabras

def sentence_similarity(vector_1: list[float], vector_2: list[float]):
    """Obtiene similitud semantica entre 2 vectores"""
    cos_sim = dot(vector_1, vector_2) / (norm(vector_1) * norm(vector_2))
    return cos_sim


