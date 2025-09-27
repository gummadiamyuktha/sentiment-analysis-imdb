import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

_stop = set(stopwords.words('english'))
_stemmer = PorterStemmer()

def clean_text(text: str, do_stem=False) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop and len(t) > 1]
    if do_stem:
        tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)
