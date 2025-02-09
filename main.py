
from gensim.models import KeyedVectors
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import Levenshtein

# Initialize components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

# Load the Word2Vec model
print("Loading the Word2Vec model...")
try:
    model = KeyedVectors.load_word2vec_format('date/GoogleNews-vectors-negative300.bin', binary=True)
    print(f"Model loaded successfully! Vocabulary size: {len(model.key_to_index)}")
except FileNotFoundError:
    print("Model file not found. Ensure 'date/GoogleNews-vectors-negative300.bin' exists.")
    exit()

def process_words(input_text):
    words = input_text.strip().split(',')
    words = [word.strip().lower() for word in words]  # Normalize input
    print(f"Input words after normalization: {words}")

    corrected_words = [spell.correction(word) for word in words]
    print(f"Words after spell correction: {corrected_words}")

    filtered_words = [word for word in corrected_words if word not in stop_words]
    valid_words = list(set(filtered_words) & set(model.key_to_index))
    print(f"Valid words in the model: {valid_words}")

    if not valid_words:
        return {
            "error": "No valid words found in the model's vocabulary.",
            "example_words": list(model.key_to_index)[:10]
        }

    vectors = np.array([model[word] for word in valid_words])

    if len(valid_words) > 1:
        centroid = np.mean(vectors, axis=0)
        TOP_N = 50
        SIMILARITY_THRESHOLD = 0.52

        most_similar_words = model.similar_by_vector(centroid, topn=TOP_N)
        print(f"Most similar words (before filtering): {most_similar_words}")
        selected_words = [
            word for word in most_similar_words
            if word[0] not in valid_words and is_different_meaning(word[0], valid_words) and word[1] > SIMILARITY_THRESHOLD
        ]

        filtered_similar_words = filter_typos(selected_words, valid_words)
        similar_words = [{"word": word, "similarity": similarity} for word, similarity in filtered_similar_words]
    else:
        similar_words = []

    return {
        "input_words": valid_words,
        "similar_words": similar_words
    }

def is_different_meaning(word, input_words):
    # Lematizare cuvintelor de intrare și ale cuvintelor similare
    input_lemmas = {lemmatizer.lemmatize(w, pos='v') for w in input_words}
    input_lemmas.update({lemmatizer.lemmatize(w, pos='n') for w in input_words})
    input_lemmas.update({w.lower() for w in input_words})

    # Verifică dacă cuvântul are aceeași rădăcină cu unul dintre cuvintele de intrare
    lemma_word = lemmatizer.lemmatize(word, pos='v')
    if lemma_word in input_lemmas:
        return False  # Cuvântul are aceeași rădăcină ca unul dintre cuvintele de intrare

    lemma_word = lemmatizer.lemmatize(word, pos='n')
    if lemma_word in input_lemmas:
        return False  # Cuvântul are aceeași rădăcină ca unul dintre cuvintele de intrare

    return word.lower() not in input_lemmas

#similaritatea Levenhstein, daca e mai mare decat threshold, este considerat typo si este exclus din lista
def filter_typos(words, valid_words, threshold=0.8):
    filtered = []
    for word, similarity in words:
        if all(Levenshtein.ratio(word, valid_word) < threshold for valid_word in valid_words):
            filtered.append((word, similarity))
    return filtered
