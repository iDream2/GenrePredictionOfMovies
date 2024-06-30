import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure required nltk data packages are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load the pre-trained models and TF-IDF vectorizer
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", 'rb'))
nb_classifier = pickle.load(open("nbmodel.pkl", 'rb'))
lr_classifier = pickle.load(open("lrmodel.pkl", 'rb'))

# Define the text preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Example new movie synopsis (change this to test with different synopses)
new_movie_synopsis = "Ten-year-old Harry Potter is an orphan who lives in the fictional London suburb of Little Whinging, Surrey, with the Dursleys: his uncaring Aunt Petunia, loathsome Uncle Vernon, and spoiled cousin Dudley. The Dursleys barely tolerate Harry, and Dudley bullies him. One day Harry is astonished to receive a letter addressed to him in the cupboard under the stairs (where he sleeps). Before he can open the letter, however, Uncle Vernon takes it. Letters for Harry subsequently arrive each day, in increasing numbers, but Uncle Vernon tears them all up, and finally, in an attempt to escape the missives, the Dursleys go to a miserable shack on a small island. On Harry’s 11th birthday, a giant named Hagrid arrives and reveals that Harry is a wizard and that he has been accepted at the Hogwarts School of Witchcraft and Wizardry. He also sheds light on Harry’s past, informing the boy that his parents, a wizard and a witch, were killed by the evil wizard Voldemort and that Harry acquired the lightning-bolt scar on his forehead during the fatal confrontation."

# Preprocess the new movie synopsis
processed_new_movie_synopsis = preprocess_text(new_movie_synopsis)

# Transform the text using the TF-IDF vectorizer
new_movie_features = tfidf_vectorizer.transform([processed_new_movie_synopsis])

# Predict the genre using the trained models
predicted_genre_nb = nb_classifier.predict(new_movie_features)
predicted_genre_lr = lr_classifier.predict(new_movie_features)

# Print the predictions
print("Predicted Genre (Naive Bayes):", predicted_genre_nb[0])
print("Predicted Genre (Logistic Regression):", predicted_genre_lr[0])
