from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    input_text = request.form.get('input_text', '')

    if not input_text:
        return render_template('index.html', error="Input text is required.")

    try:
        # Call the backend API
        response = requests.post('http://127.0.0.1:5000/process', json={"text": input_text})
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        result = response.json()
        if "error" in result:
            return render_template('index.html', error=result["error"], input_text=input_text)

        return render_template('index.html', result=result.get("similar_words", []), input_text=input_text)
    except requests.exceptions.RequestException as e:
        return render_template('index.html', error=f"Error connecting to backend: {e}")
    except ValueError:
        return render_template('index.html', error="Invalid response from backend. Ensure it returns JSON.")


if __name__ == '__main__':
    app.run(debug=True, port=5001)
