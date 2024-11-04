from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def encode_text(text):
    # Tokenize and encode text to get DistilBERT embeddings
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
    
    return embeddings


def calculate_similarity(text1, text2):
    # Encode both texts
    embedding1 = encode_text(text1)
    embedding2 = encode_text(text2)
    
    # Convert embeddings to numpy arrays and compute cosine similarity
    embedding1 = embedding1.detach().numpy()
    embedding2 = embedding2.detach().numpy()
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text1 = data.get('text1')
    text2 = data.get('text2')
    

    if not text1 or not text2:
        return jsonify({"error": "Both text1 and text2 are required"}), 400
    
    similarity_score = calculate_similarity(text1, text2)
    

    return jsonify({
        "text1": text1,
        "text2": text2,
        "similarity_score": similarity_score
    })



if __name__ == '__main__':
    app.run(debug=True)
