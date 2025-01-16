import nltk  
import numpy as np  
import random  
import string

# Importing libraries for web scraping and text processing
import bs4 as bs  
import urllib.request  
import re  

# Importing stopwords and downloading required NLTK resources
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('rslp')

from string import punctuation

# Importing heapq for identifying most/least frequent words
import heapq

# Importing NLTK tokenizers
from nltk.tokenize import word_tokenize, sent_tokenize

# Portuguese stemmer
stemmer = nltk.stem.RSLPStemmer()

# Function for stemming
def Stemming(text):
    """
    Performs stemming on a list of tokens using the RSLPStemmer.
    Args:
        text (list): List of tokens (words).
    Returns:
        list: List of stemmed tokens.
    """
    stemmer = nltk.stem.RSLPStemmer()
    new_text = []
    for token in text:
        new_text.append(stemmer.stem(token.lower()))
    return new_text

# URL of the text file
urlPath = 'https://learn-us-east-1-prod-fleet01-xythos.s3.amazonaws.com/5df7dfcfaf23d/102782?response-cache-control=private%2C%20max-age%3D21600&response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%27corpus30NoticiasCurtas.txt&response-content-type=text%2Fplain&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200609T000000Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21600&X-Amz-Credential=AKIAZH6WM4PLTYPZRQMY%2F20200609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=7f1980a42b5921f94fd23ca5bdad4b56895eb4b6b8cf3e6ea46841d278fa5097'

# Reading raw HTML from the URL
raw_html = urllib.request.urlopen(urlPath)  
raw_html = raw_html.read()

# Parsing the HTML using BeautifulSoup
article_html = bs.BeautifulSoup(raw_html, 'lxml')

# Extracting paragraphs from the HTML
article_paragraphs = article_html.find_all('p')

# Concatenating all paragraphs into a single string
article_text = ''
for para in article_paragraphs:  
    article_text += para.text

# Splitting the text into sentences
frases = article_text.split('\r\n')[:30]
corpus = nltk.sent_tokenize(article_text)

# Convert to lowercase, remove punctuation, and clean whitespace
for i in range(len(corpus)):
    corpus[i] = corpus[i].lower()
    corpus[i] = re.sub(r'\W', ' ', corpus[i])
    corpus[i] = re.sub(r'\s+', ' ', corpus[i])

# Define Portuguese stopwords and add punctuation
stopwords = set(stopwords.words('portuguese') + list(punctuation))

# Create a Bag of Words with word frequency
wordfreq = {}
for sentence in corpus:
    tokens = word_tokenize(sentence, language='portuguese')

    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords]

    # Apply stemming
    tokens = Stemming(tokens)

    # Count word frequency
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

# Identify the 10 most and least frequent words
most_freq = heapq.nlargest(10, wordfreq, key=wordfreq.get)
less_freq = heapq.nsmallest(10, wordfreq, key=wordfreq.get)

# Print results
print('\n10 Most Frequent Words: {}'.format(most_freq))
print('\n10 Least Frequent Words: {}'.format(less_freq))
