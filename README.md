# Projet de Text Mining et Analyse de Sentiments

Ce projet est une application de **text mining** et **analyse de sentiments** développée en Python. Il utilise des techniques de traitement du langage naturel (NLP) pour analyser le sentiment (positif, neutre, négatif) de textes en anglais et en français. Le modèle est basé sur l'algorithme **K-Nearest Neighbors (KNN)** et utilise **TF-IDF** pour la vectorisation des textes.

## Fonctionnalités

- **Prétraitement des textes** : Nettoyage des textes (suppression des caractères spéciaux, tokenization, suppression des stopwords, lemmatisation).
  
- **Analyse de sentiments** : Classification des textes en trois catégories : positif, neutre, négatif.
  
- **Prédiction de nouveaux textes** : Possibilité de prédire le sentiment de nouveaux textes en utilisant le modèle entraîné.

## Technologies utilisées

- **Python** : Langage de programmation principal.
  
- **Pandas** : Pour la manipulation des données.
  
- **Scikit-learn** : Pour la vectorisation des textes (TF-IDF) et le modèle KNN.
  
- **NLTK** : Pour le prétraitement des textes (tokenization, stopwords, lemmatisation).

## Structure du projet

- **`text_mining.py`** : Script principal contenant le code pour le prétraitement des textes, l'entraînement du modèle, et la prédiction des sentiments.
  
- **`requirements.txt`** : Fichier listant les dépendances nécessaires pour exécuter le projet.

## Installation

1. **Cloner le dépôt** :

   git clone https://github.com/votre-utilisateur/text-mining-project.git
   
   cd text-mining-project


2. **Installer les dépendances** :

    Assurez-vous d'avoir Python 3 installé, puis exécutez la commande suivante pour installer les dépendances :

   pip install -r requirements.txt


3. **Télécharger les données NLTK** :

   
   Le script nécessite des données supplémentaires de NLTK. Exécutez la commande suivante pour les télécharger :
  
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"


4. **Exécuter le projet** :
   
   Exécutez le script principal pour entraîner le modèle et effectuer des prédictions :
   
   python text_mining.py





