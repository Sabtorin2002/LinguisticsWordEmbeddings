# LinguisticsWordEmbeddings

# Word Similarity Finder

## Overview
The **Word Similarity Finder** application is designed to process input words, find their valid representations in a pre-trained Word2Vec model, and suggest similar words based on semantic similarity. The project utilizes Natural Language Processing (NLP) techniques to provide meaningful word associations.

## Features
- Processes input words and normalizes them.
- Performs spell-checking and stopword removal.
- Retrieves similar words using a **pre-trained Word2Vec model**.
- Additional filtering using lemmatization and typo detection.

## Technologies Used
### Libraries
- **Gensim**: Handles Word2Vec models and vector operations.
- **NumPy**: Supports numerical computations.
- **NLTK (Natural Language Toolkit)**: Provides NLP preprocessing tools.
- **SpellChecker**: Corrects spelling errors.
- **Levenshtein**: Computes string similarity for typo filtering.

### Model Used
- **GoogleNews-vectors-negative300.bin**: A pre-trained Word2Vec model trained on the Google News corpus (~100 billion words) with 300-dimensional word embeddings.

## Methodology
### Input Processing:
1. **Normalization**: Converts input words to lowercase and removes extra spaces.
2. **Spell-checking**: Corrects misspelled words.
3. **Stopword Removal**: Filters out common words.
4. **Validation**: Retains words present in the Word2Vec vocabulary.

### Similarity Calculation:
1. Computes the **centroid** of valid word embeddings.
2. Retrieves the top 50 most similar words from the model.

### Additional Features:
- **Lemmatization**: Ensures words are in their root form.
- **Typo Filtering**: Uses Levenshtein similarity to remove overly similar words.

## Backend
- **Flask-based API**
- **Request method**: `POST`
- **Processing Flow**:
  1. Accepts JSON input: `{ "text": "king, woman" }`
  2. Normalizes, corrects, and validates input words.
  3. Computes similar words using Word2Vec.
  4. Returns JSON response:
     ```json
     {
       "similar_words": [
         {"word": "prince", "similarity": 0.653},
         {"word": "princess", "similarity": 0.611}
       ]
     }
     ```

## Frontend
- **Flask-based web interface**
- **User Input Handling**:
  - Accepts text input via form submission.
  - Sends a `POST` request to the API.
  - Displays results or error messages.

## Example Results
| Input Words       | Similar Words Output                        |
|------------------|-------------------------------------------|
| king, woman      | prince (0.653), princess (0.611)          |
| beer, wine, vodka| whiskey (0.724), rum (0.689), gin (0.650) |
| chocolate, ice   | cream (0.745), caramel (0.702)            |

## Conclusion
The **Word Similarity Finder** efficiently generates semantically related words using Word2Vec embeddings. The application combines text preprocessing, NLP techniques, and a simple Flask-based API to deliver meaningful results.

## Installation & Usage
### Prerequisites:
- Python 3.x
- Install dependencies:
  ```bash
  pip install gensim numpy nltk flask pyspellchecker python-Levenshtein
  ```
### Running the Application:
1. **Start the Backend API**:
   ```bash
   python app.py
   ```
2. **Access the Frontend**:
   Open `http://localhost:5000` in a web browser and enter words to find similar ones.

## License
This project is licensed under the MIT License.

