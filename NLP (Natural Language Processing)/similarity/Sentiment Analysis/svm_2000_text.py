# -*- coding: utf-8 -*-
"""
This script performs sentiment analysis using an SVM (Support Vector Machine) classifier.
It processes textual data, creates a Bag of Words (BoW) feature matrix, and evaluates the model using cross-validation.
"""

from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from operator import itemgetter, attrgetter
import numpy as np
import nltk
import math

# Download necessary NLTK resources
nltk.download(['stopwords', 'rslp', 'punkt'])

# Function to train and evaluate SVM
def SVM(features, labels, folds):
    """
    Trains an SVM classifier using linear kernel and evaluates using cross-validation.
    
    Args:
        features (list): Feature matrix.
        labels (list): Labels for classification.
        folds (int): Number of folds for cross-validation.

    Returns:
        tuple: Predicted labels and confusion matrix.
    """
    print('Linear SVM start:')
    clf = svm.SVC(kernel='linear', C=1)
    predict = cross_val_predict(clf, features, labels, cv=folds)
    cm = confusion_matrix(labels, predict)
    print(cm)
    print(classification_report(labels, predict, target_names=np.unique(labels)))
    print('----------------------------------------------')
    return predict, cm

# Function to load and process input file
def loadFile(filename, stopwords, stemmer):
    """
    Loads a file, tokenizes its text, removes stopwords, and applies stemming. 
    Generates a Bag of Words (BoW) feature matrix.
    
    Args:
        filename (str): Path to the input file.
        stopwords (list): List of stopwords to remove.
        stemmer: Stemmer to reduce words to their root forms.

    Returns:
        tuple: Labels, feature matrix, and list of unique words.
    """
    f = open(filename, "r", encoding="utf-8-sig")
    data = []
    while True:
        chunk = f.readline()  # Reads a line from the file
        if chunk == '':  # Checks if the file has ended
            break
        text = chunk.split(';;')  # Splits into class (text[0]) and text content (text[1])
        
        tokens = nltk.word_tokenize(text[1])  # Tokenizes the line
        tokens_clean = []
        for word in tokens:
            if word not in stopwords:
                tokens_clean.append(stemmer.stem(word))  # Adds stemmed word to the clean token list
        instance = (text[0], tokens_clean)  # Creates an instance with class and tokens
        data.append(instance)

    # Bag of Words (BoW) generation
    # Lists unique tokens
    allwords = []
    for text in data:
        for word in text[1]:
            if word not in allwords:
                allwords.append(word)
    allwords.sort()  # Sorts tokens alphabetically

    # Generates occurrence matrix for words
    qtdPalavras = len(allwords)
    features = []
    labels = []
    for text in data:
        occurrences = []
        labels.append(text[0])
        for word in allwords:
            occurrences.append(text[1].count(word)) if word in text[1] else occurrences.append(0)
        features.append(occurrences)

    return labels, features, allwords

# Stopwords list
# Loads Portuguese stopwords and extends with additional symbols and vowels
stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords += (',', '.', '(', ')', '"', "'", '´', '`', '!', '$', '%', '&', '...', '-', ':', ';', '?', '``', '\'\'')
stopwords += ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')

# Portuguese stemmer
stemmer = nltk.stem.RSLPStemmer()

# Process dataset with 7 classes
InputFile = r".\2000_textos_minusculo.txt"
labels7, features7, allwords7 = loadFile(InputFile, stopwords, stemmer)

print('----------------------------------------------')
print("Instances with 7 classes: " + str(len(labels7)))
print("Features with 7 classes: " + str(len(features7[0])))

predict7, cm7 = SVM(features7, labels7, 10)

# Process dataset with 3 classes
InputFile = "1748_textos_polarizado.txt"
labels3, features3, allwords3 = loadFile(InputFile, stopwords, stemmer)

print('----------------------------------------------')
print("Instances with 3 classes: " + str(len(labels3)))
print("Features with 3 classes: " + str(len(features3[0])))

predict3, cm3 = SVM(features3, labels3, 10)
