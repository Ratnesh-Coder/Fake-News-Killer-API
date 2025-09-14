from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests
from bs4 import BeautifulSoup
from googlesearch import search

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Allow cross-origin requests from the browser extension

# --- Load the trained model and vectorizer ---
# These files must be in the same directory as this script
try:
    model = joblib.load('fake_news_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("✅ Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("🔴 Error: Model or vectorizer files not found. Make sure they are in the same directory.")
    exit()

def cross_reference_online(text):
    """
    Performs a web search and scrapes the top results to find sources.
    """
    print(f"🔎 Cross-referencing text online: '{text[:50]}...'")
    sources = []
    try:
        # Use googlesearch to find the top 3 relevant URLs
        search_results = search(text, num_results=3, lang="en")
        
        for url in search_results:
            try:
                # Make a request to the URL with a timeout and user-agent
                response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status() # Raise an exception for bad status codes

                # Parse the HTML to get the page title
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.string if soup.title else "No title found"
                
                sources.append({
                    "title": title.strip(),
                    "url": url
                })
                print(f"  - Found source: {title.strip()}")

            except requests.RequestException as e:
                print(f"  - ⚠️ Could not fetch URL {url}: {e}")
                continue # Skip to the next URL

    except Exception as e:
        print(f"🔴 An error occurred during web search: {e}")

    return sources


@app.route('/predict', methods=['POST'])
def predict():
    """
    The main API endpoint. It gets a prediction from the ML model
    and then cross-references the text online.
    """
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Missing "text" in request body'}), 400

    text_to_check = request.json['text']

    # --- Step 1: Get initial prediction from the local ML model ---
    vectorized_text = vectorizer.transform([text_to_check])
    prediction = model.predict(vectorized_text)[0]
    print(f"🧠 Model prediction: {prediction}")

    # --- Step 2: Get online sources ---
    online_sources = cross_reference_online(text_to_check)

    # --- Step 3: Combine and return the results ---
    response_data = {
        'prediction': prediction,
        'sources': online_sources
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    # Use 0.0.0.0 to make it accessible on your local network if needed
    app.run(host='127.0.0.1', port=5000, debug=True)

