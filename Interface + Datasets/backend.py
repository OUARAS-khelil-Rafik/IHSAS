"""
Installation les bibliothèque
pip install aaransia

pip install nltk

pip install tashaphyne

pip install --upgrade arabic-reshaper
pip install wordcloud
"""

import nltk
"""
nltk.download('stopwords')
nltk.download('punkt')
"""
from flask import Flask, request, jsonify, render_template, send_file
from io import BytesIO
from sklearn.naive_bayes import  MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="http://127.0.0.1:5501") 

test = pd.read_csv("./Dataset/testing_data.csv")
train = pd.read_csv("./Dataset/training_data.csv")

data = pd.concat([train, test], ignore_index=True)

# Mock training data (replace it with your actual training data)
X_train = data["tweet"]
y_train = data["sentiment"]

# Transformation Aransia en dialectique :

import warnings
from aaransia import transliterate, SourceLanguageError

warnings.simplefilter('ignore')

for i in range(len(data)):
    row = data.iloc[i]["tweet"]
    try:
        row = transliterate(row, source='ar', target='ar', universal=True)
        data.loc[:, "tweet"][i] = row
    except SourceLanguageError as e:
        print(f"Erreur de translittération à l'index {i}: {e}")

# Normalisation :

import re

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

# Applique la normalisation aux tweets
data['tweet'] = data['tweet'].apply(normalize_arabic)

# Nettoyage des données

import re

# Remove hyperlinks
data['tweet'] = [re.sub(r'http\S+ | htps\S+', '', str(s)) for s in data['tweet']]

# Handle URLs
data['tweet'] = [re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', str(s)) for s in data['tweet']]

# Remove words starting with @
data['tweet'] = [re.sub(r'@\S+', '', str(s)) for s in data['tweet']]

# Remove # and _
data['tweet'] = data['tweet'].str.replace("_", " ").str.replace("#", "")

data['tweet'] = data['tweet'].str.replace('. | , | ، | ؛', ' ')

# Handle Twitter reserved words
data['tweet'] = [re.sub(r'\bRT\b | \bRetweeted\b', '', str(s)) for s in data['tweet']]

data['tweet'] = [re.sub(r'[\u0660-\u0669\u06F0-\u06F9]+', '', str(s)) for s in data['tweet']]

# Remove consecutive duplicate characters
data['tweet'] = [re.sub(r"(.)\1{2,}", r"\1", str(s)) for s in data['tweet']] if len(data['tweet']) > 2 else data['tweet'][:2]

data['tweet'] = data['tweet'].str.replace('\d+', ' ')
data['tweet'] = data['tweet'].str.replace('\n', ' ')
data['tweet'] = data['tweet'].str.replace('/', ' ')
data['tweet'] = [re.sub(r'[^\w\s]', '', str(s)) for s in data['tweet']]

# Remplace les valeurs nulles par une chaîne de caractères vide
data['tweet'] = data['tweet'].fillna('')

# Stop-Words :

# import necessary libraries
from nltk.corpus import stopwords

stopwords = list(set(nltk.corpus.stopwords.words('arabic')))
data["tweet"] = data["tweet"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

# Stemming :

from tashaphyne.stemming import ArabicLightStemmer

# Stemmer_LIGHT : Supprimer "suffixes" et "affixes" 
ArListem = ArabicLightStemmer()
def stemmer_light(text):
    text_words = []
    words = text.split(" ")
    for c in words:
        stem = ArListem.light_stem(c)
        text_words.append(stem)
    return ' '.join(text_words)

# Root Stemming : Transformer le mot dans sa forme racine
def stemmer_root(text):
    text_words = []
    words = text.split(" ")
    for c in words:
        stem = ArListem.light_stem(c)
        text_words.append(stem)
    return ' '.join(text_words)

sentences = [stemmer_light(text) for text in data['tweet']]

data['tweet'] = sentences

# Nous devons créer une matrice de termes de document en utilisant CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in  vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# Word Cloud :

import nltk
import arabic_reshaper

arab_stopwords = list(set(nltk.corpus.stopwords.words("arabic")))
sample_corpus=' '.join(data['tweet'])
data_arb = arabic_reshaper.reshape(sample_corpus)

# Classificateur d'analyse des sentiments :
# Création d'un ensemble de données d'entraînement pour le classificateur :

# Tokenisation :

import nltk

from nltk.tokenize import word_tokenize
tokenizedWords = []
documents = []

# Chaque document contient tuple ==> la liste de mot et categorie
for i in data.index:
    sentiment = data["sentiment"][i]
    review = data["tweet"][i]
    tokenizedWord = word_tokenize(review)
    document = [tokenizedWord, sentiment]
    documents.append(document)

# Liste tous les mots :

listeall=[]

for i in data["tweet"]:
    review = i
    tokenizedWord = word_tokenize(review)
    for j in tokenizedWord:
        listeall.append(j)

# Bag of word (Sac de mots) :

# Définir l'extracteur de fonctionnalités
# Permet de calculer la frequence de chaque mot dans le document
all_words = nltk.FreqDist(w.lower() for w in listeall)
word_features = list(all_words)[:2000]
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

df_fdist = pd.DataFrame.from_dict(all_words, orient='index')
df_fdist.columns = ['Frequency']
df_fdist.index.name = 'Term'

# Model :

vectorizer = CountVectorizer(max_features=8000, ngram_range=(1,3))
X_train_vectorized = vectorizer.fit_transform(X_train) 

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.get_json()
    input_text = data['text']

    input_vectorized = vectorizer.transform([input_text])
    prediction = model.predict(input_vectorized)[0]

    # Map the abbreviated sentiment to full words
    sentiment_mapping = {'POS': 'Positive', 'NEG': 'Negative', 'NEU': 'Neutral'}
    full_sentiment = sentiment_mapping.get(prediction, 'Unknown')

    return jsonify({'result': full_sentiment})

@app.route('/classify-and-download', methods=['POST'])
def classify_and_download():
    file = request.files['file']

    if file and file.filename.endswith('.txt'):
        # Read the content of the file
        content = file.read().decode('utf-8').splitlines()

        # Classify sentiments
        X_vectorized = vectorizer.transform(content)
        predictions = model.predict(X_vectorized)

        # Create a DataFrame with original tweets and predicted sentiment
        result_df = pd.DataFrame({'tweet': content, 'sentiment': predictions})

        # Save the DataFrame to a CSV file in-memory using BytesIO
        csv_output = BytesIO()
        result_df.to_csv(csv_output, index=False, encoding='utf-8')
        csv_output.seek(0)

        # Send the CSV file as a response
        return send_file(csv_output, mimetype='text/csv', as_attachment=True, download_name='classified_tweets.csv')

    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
