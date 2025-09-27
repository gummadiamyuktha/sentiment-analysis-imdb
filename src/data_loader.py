import nltk
from nltk.corpus import movie_reviews
import numpy as np

def ensure_nltk_data():
    try:
        nltk.data.find('corpora/movie_reviews')
    except LookupError:
        nltk.download('movie_reviews')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def load_movie_reviews():
    ensure_nltk_data()
    texts, labels = [], []
    for fid in movie_reviews.fileids('pos'):
        texts.append(movie_reviews.raw(fid))
        labels.append(1)
    for fid in movie_reviews.fileids('neg'):
        texts.append(movie_reviews.raw(fid))
        labels.append(0)
    rng = np.random.default_rng(42)
    idx = np.arange(len(texts))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = [int(labels[i]) for i in idx]
    return texts, labels
