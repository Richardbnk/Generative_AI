# -*- coding: utf-8 -*-
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import numpy as np
import nltk
import math
import re
import matplotlib.pyplot as plt

# Download necessary NLTK resources
nltk.download(['stopwords', 'rslp', 'punkt'])

# Function to train and evaluate an SVM model
def SVM(features, labels, folds):
    """
    Trains an SVM model with a linear kernel and evaluates it using cross-validation.
    
    Args:
        features (list): Feature matrix.
        labels (list): Labels for classification.
        folds (int): Number of folds for cross-validation.

    Returns:
        tuple: Predicted labels, confusion matrix and trained model.
    """
    print('Linear SVM start: ')
    clf = svm.SVC(kernel='linear', C=1)
    clf = clf.fit(features, labels)  # Train the model
    predict = cross_val_predict(clf, features, labels, cv=folds)  # Cross-validation
    matrix = confusion_matrix(labels, predict)  # Confusion matrix
    print(matrix)
    print(classification_report(labels, predict, target_names=np.unique(labels)))
    print('----------------------------------------------')
    return predict, matrix, clf

# Function to load and preprocess text data
def loadFile(filename, stopwords, stemmer):
    """
    Loads text data, processes it, and creates a Bag of Words (BoW) model.
    
    Args:
        filename (str): Path to the text file.
        stopwords (list): List of stopwords to remove.
        stemmer: Stemmer to apply to tokens.

    Returns:
        tuple: Labels, feature matrix, and list of unique words.
    """
    f = open(filename, "r", encoding="utf-8-sig")
    data = []
    while True:
        chunk = f.readline()  # Read a line from the file
        if chunk == '':  # Check if end of file is reached
            break

        # Exclude lines labeled as "surpresa" (surprise)
        if not ('surpresa' in chunk):
            text = chunk.split(';;')  # Split into class (text[0]) and content (text[1])

            # Preprocess: lowercase, remove punctuation and extra spaces
            for i in range(len(text)):
                text[i] = text[i].lower()
                text[i] = re.sub(r'\W', ' ', text[i])
                text[i] = re.sub(r'\s+', ' ', text[i])

            # Tokenize and clean the text
            tokens = nltk.word_tokenize(text[1])
            tokens_clean = [stemmer.stem(word) for word in tokens if word not in stopwords]

            # Map original classes to broader classes (positive/negative)
            newText = text[0]
            newText = newText.replace('alegria', 'positivo')
            newText = newText.replace('raiva', 'negativo')
            newText = newText.replace('medo', 'negativo')
            newText = newText.replace('desgosto', 'negativo')
            newText = newText.replace('tristeza', 'negativo')

            # Create an instance with the new label and clean tokens
            instance = (newText, tokens_clean)
            data.append(instance)

    # Create a Bag of Words (BoW) model
    allwords = sorted(set(word for text in data for word in text[1]))  # Unique tokens
    features = []
    labels = []
    for text in data:
        labels.append(text[0])  # Add label
        occurrences = [text[1].count(word) for word in allwords]  # Word occurrences
        features.append(occurrences)

    return labels, features, allwords

# Define stopwords and stemmer
stopwords = nltk.corpus.stopwords.words('portuguese')  # Load Portuguese stopwords
stopwords += (',', '.', '(', ')', '"', "'", '´', '`', '!', '$', '%', '&', '...', '-', ':', ';', '?', '``', '\'\'')  # Add symbols
stopwords += ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')  # Add vowels
stemmer = nltk.stem.RSLPStemmer()  # Portuguese stemmer

# Load and preprocess text data
filename = r".\2000_textos.txt"
labels3, features3, allwords3 = loadFile(filename, stopwords, stemmer)

print('----------------------------------------------') 
print("Instances (3 classes): " + str(len(labels3)))
print("Features (3 classes): " + str(len(features3[0])))

# Train and evaluate the SVM model
predict3, matrix3, clf = SVM(features3, labels3, 10)

# Plot confusion matrices (with and without normalization)
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, features3, labels3, normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
