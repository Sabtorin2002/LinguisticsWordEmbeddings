from flask import Flask, request, jsonify
from main import process_words

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process_input():
    try:
        data = request.get_json()
        input_text = data.get('text', '')

        if not input_text:
            return jsonify({"error": "Input text is required"}), 400

        result = process_words(input_text)
        similar_words = result.get("similar_words", [])
        return jsonify({"similar_words": similar_words})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
