import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Données intégrées
data = [
    {"text_content": "The product exceeded my expectations! Excellent quality.", "sentiment_label": "positive"},
    {"text_content": "I am disappointed with the quality of this item.", "sentiment_label": "negative"},
    {"text_content": "This is an average product. Nothing special.", "sentiment_label": "neutral"},
    {"text_content": "J'adore ce produit! La qualité est excellente.", "sentiment_label": "positive"},
    {"text_content": "Produit inutile, ne correspond pas à la description.", "sentiment_label": "negative"},
    {"text_content": "The product is functional and does the job.", "sentiment_label": "neutral"},
    {"text_content": "Amazing purchase! Highly recommend.", "sentiment_label": "positive"},
    {"text_content": "The build quality is poor. Waste of money.", "sentiment_label": "negative"},
    {"text_content": "It's an okay product. Could be better.", "sentiment_label": "neutral"},
    {"text_content": "Le produit est vraiment parfait, je l'adore!", "sentiment_label": "positive"},
]

# Charger les données dans un DataFrame
df = pd.DataFrame(data)

# Get stopwords in both English and French
stop_words_english = set(stopwords.words('english'))
stop_words_french = set(stopwords.words('french'))
all_stop_words = stop_words_english.union(stop_words_french)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Nettoyage et prétraitement des textes amélioré
def preprocess_text(text):
    # Supprimer les caractères spéciaux et les chiffres
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)
    
    # Convertir en minuscules
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Supprimer les stopwords et appliquer la lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
             if word not in all_stop_words and len(word) > 2]
    
    return ' '.join(tokens)

# Appliquer le nettoyage
df['cleaned_text'] = df['text_content'].apply(preprocess_text)

# Transformer les étiquettes en valeurs numériques
label_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
df['label'] = df['sentiment_label'].map(label_mapping)

# Séparer les données en caractéristiques (X) et cibles (y)
X = df['cleaned_text']
y = df['label']

# Vectorisation des textes avec TF-IDF
vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
X_vectorized = vectorizer.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# Modèle KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = knn.predict(X_test)

# Afficher les résultats
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Fonction pour prédire le sentiment de nouveaux textes
def predict_sentiment(texts):
    cleaned_texts = [preprocess_text(text) for text in texts]
    vectorized_texts = vectorizer.transform(cleaned_texts)
    predictions = knn.predict(vectorized_texts)
    return predictions

# Exemple de prédiction sur de nouveaux textes
new_texts = [
    "The product is amazing! Highly recommend it.",
    "Not worth the money. Very poor quality.",
    "It does what it says, nothing more, nothing less."
]

# Prédictions
predictions = predict_sentiment(new_texts)

# Afficher les résultats
print("\nPredictions on new texts:")
for text, pred in zip(new_texts, predictions):
    sentiment = {1: "positive", 0: "neutral", -1: "negative"}[pred]
    print(f"Text: {text}\nSentiment: {sentiment}\n")