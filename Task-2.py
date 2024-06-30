import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the dataset
df = pd.read_csv("./Genre Classification Dataset/train_data.txt", sep=":::", header=None, names=["Movie", "Genre", "Synopsis"], engine='python')

# Drop missing values
df = df.dropna(subset=['Synopsis'])

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(df.head())
print(df.info())
print(df.shape)

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing to the synopsis column
df['Processed_Synopsis'] = df['Synopsis'].apply(preprocess_text)

print(df[['Synopsis', 'Processed_Synopsis']].head())

# Convert text data into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['Processed_Synopsis'])
pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pkl", 'wb'))

# Target variable
y = df['Genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
pickle.dump(nb_classifier, open("nbmodel.pkl", 'wb'))

# Train a Logistic Regression classifier
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train, y_train)
pickle.dump(lr_classifier, open("lrmodel.pkl", 'wb'))

# Evaluate Naive Bayes classifier
nb_predictions = nb_classifier.predict(X_test)
print("Naive Bayes Classifier Report:")
print(classification_report(y_test, nb_predictions))
print("Accuracy:", accuracy_score(y_test, nb_predictions))

# Evaluate Logistic Regression classifier
lr_predictions = lr_classifier.predict(X_test)
print("Logistic Regression Classifier Report:")
print(classification_report(y_test, lr_predictions))
print("Accuracy:", accuracy_score(y_test, lr_predictions))

# Example new movie synopsis
new_movie_synopsis = "A young wizard embarks on a journey to defeat a dark lord."

# Preprocess the new movie synopsis
processed_new_movie_synopsis = preprocess_text(new_movie_synopsis)

# Transform the text using the same TF-IDF vectorizer
new_movie_features = tfidf_vectorizer.transform([processed_new_movie_synopsis])

# Predict the genre using the trained model
predicted_genre_nb = nb_classifier.predict(new_movie_features)
predicted_genre_lr = lr_classifier.predict(new_movie_features)

print("Predicted Genre (Naive Bayes):", predicted_genre_nb[0])
print("Predicted Genre (Logistic Regression):", predicted_genre_lr[0])
