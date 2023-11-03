from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, render_template, jsonify
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import pandas as pd
import string
import nltk

app = Flask(__name__, template_folder='templates')

# Your existing code for loading data and setting up the classifier...

# Define the function for tokenization and preprocessing text
def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

# Assuming 'dataset.csv' is your actual data file and 'text' and 'spam' are columns in your CSV
df = pd.read_csv('dataset.csv')
df.drop_duplicates(inplace=True)
nltk.download('stopwords')

count_vectorizer = CountVectorizer(analyzer=process_text)
messages_bow = count_vectorizer.fit_transform(df['text'])

X = messages_bow
y = df['spam']

classifier = MultinomialNB(alpha=1.0)
classifier.fit(X, y)

# Function to classify a new text as spam or not spam
def classify_text(text):
    processed_text = process_text(text)
    vectorized_text = count_vectorizer.transform([text])
    prediction = classifier.predict(vectorized_text)
    return prediction[0]

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        text = data['text']
        prediction = classify_text(text)
        result = {'prediction': int(prediction)}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
