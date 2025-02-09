from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import Levenshtein  # For string similarity checks
import spacy
# Download necessary NLTK data
# Uncomment these if running for the first time
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize stopwords, lemmatizer, and spell checker
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

# Function to check if a word has a different meaning
def is_different_meaning(word, input_words):
    input_lemmas = {lemmatizer.lemmatize(w, pos='v') for w in input_words}
    input_lemmas.update({lemmatizer.lemmatize(w, pos='n') for w in input_words})
    input_lemmas.update({w.lower() for w in input_words})

    return (
        lemmatizer.lemmatize(word, pos='v') not in input_lemmas and
        lemmatizer.lemmatize(word, pos='n') not in input_lemmas and
        word.lower() not in input_lemmas
    )

# Function to filter out typo-like results based on Levenshtein distance
def filter_typos(words, valid_words, threshold=0.8):
    filtered = []
    for word, similarity in words:
        # Check the similarity of the candidate word with each valid input word
        if all(Levenshtein.ratio(word, valid_word) < threshold for valid_word in valid_words):
            filtered.append((word, similarity))
    return filtered

# Load the pre-trained Word2Vec model
print("Loading the Word2Vec model...")
try:
    model = KeyedVectors.load_word2vec_format('date/GoogleNews-vectors-negative300.bin', binary=True)
    print(f"Model loaded successfully! Vocabulary size: {len(model.key_to_index)}")
except FileNotFoundError:
    print("Model file not found. Ensure 'data/GoogleNews-vectors-negative300.bin' exists.")
    exit()

# Accept user input
words = input("Your words:").strip().split(',')
words = [word.strip().lower() for word in words]  # Normalize input
print("Original input words:", words)

# Spell check and correct words
corrected_words = [spell.correction(word) for word in words]
print("Words after spell checking:", corrected_words)

# Remove stopwords
filtered_words = [word for word in corrected_words if word not in stop_words]
print("Words after removing stopwords:", filtered_words)

# Check valid words in the vocabulary
valid_words = list(set(filtered_words) & set(model.key_to_index))
if not valid_words:
    print("No valid words found. Ensure the words are in the model's vocabulary.")
    print("Example words from vocabulary:", list(model.key_to_index)[:10])
else:
    print(f"Valid words: {valid_words}")

    # Get vectors for the valid words
    vectors = np.array([model[word] for word in valid_words])

    # Find the words most similar to the input words
    if len(valid_words) > 1:
        # Calculate the centroid of the valid word vectors
        centroid = np.mean(vectors, axis=0)

        TOP_N = 50  # Number of similar words to return
        SIMILARITY_THRESHOLD = 0.52  # Similarity threshold

        # Find the most similar words to the centroid excluding the input words and their derivatives
        most_similar_words = model.similar_by_vector(centroid, topn=TOP_N)
        selected_words = [
            word for word in most_similar_words
            if word[0] not in valid_words and is_different_meaning(word[0], valid_words) and word[1] > SIMILARITY_THRESHOLD
        ]

        # Filter out typo-like results
        filtered_similar_words = filter_typos(selected_words, valid_words)

        if filtered_similar_words:
            print("Words most similar (excluding input words and derivatives):")
            for word, similarity in filtered_similar_words:
                print(f"{word} (similarity: {similarity:.4f})")
        else:
            print("No similar words found with similarity above 0.5, excluding input words and derivatives.")
    else:
        print("Cannot calculate a representative word with less than two valid words.")

    # Reduce dimensions using t-SNE
    perplexity = min(30, len(valid_words) - 1)
    print("Reducing dimensions with t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    print("Reduced vectors shape:", reduced_vectors.shape)

    # Plot the 2D visualization
    print("Plotting the words...")
    plt.figure(figsize=(8, 8))
    for i, word in enumerate(valid_words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], label=word)
        plt.text(reduced_vectors[i, 0] + 0.01, reduced_vectors[i, 1] + 0.01, word, fontsize=9)

    plt.title("Word Embeddings Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()