import nltk  
import numpy as np  
import random  
import string

# Libraries for web scraping and data processing
import bs4 as bs  
import urllib.request  
import re  

# Libraries for visualization and mathematical functions
import matplotlib.pyplot as plt
from scipy import special

# Tokenization and N-Gram utilities from NLTK
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

# Download NLTK resources
nltk.download('punkt')

# Function to extract character-based N-Grams
def extrair_ngrams_chars(text, n):
    """
    Extracts N-Grams from the input text based on characters.
    
    Args:
        text (str): The input text.
        n (int): The size of the N-Grams.
        
    Returns:
        list: A list of character-based N-Grams.
    """
    chars = [c for c in text]
    n_grams = ngrams(chars, n) 
    return [''.join(grams) for grams in n_grams]

# Load text from the specified URL
urlPath = 'https://learn-us-east-1-prod-fleet01-xythos.s3.amazonaws.com/5df7dfcfaf23d/102782?response-cache-control=private%2C%20max-age%3D21600&response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%27corpus30NoticiasCurtas.txt&response-content-type=text%2Fplain&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200609T000000Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21600&X-Amz-Credential=AKIAZH6WM4PLTYPZRQMY%2F20200609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=7f1980a42b5921f94fd23ca5bdad4b56895eb4b6b8cf3e6ea46841d278fa5097'
raw_html = urllib.request.urlopen(urlPath)  
raw_html = raw_html.read()

# Parse the HTML and extract text
article_html = bs.BeautifulSoup(raw_html, 'lxml')
article_paragraphs = article_html.find_all('p')

# Combine all paragraphs into a single text
article_text = ''
for para in article_paragraphs:  
    article_text += para.text

# Split the text into sentences
frases = article_text.split('\r\n')[:30]
corpus = nltk.sent_tokenize(article_text)

# Preprocess the corpus: lowercase, remove punctuation, and extra spaces
for i in range(len(corpus)):
    corpus[i] = corpus[i].lower()
    corpus[i] = re.sub(r'\W', ' ', corpus[i])
    corpus[i] = re.sub(r'\s+', ' ', corpus[i])

# Function to extract unique N-Grams
def uniqueNgrams(ngrams):
    """
    Extracts unique N-Grams from the input list.
    
    Args:
        ngrams (list): List of N-Grams.
        
    Returns:
        list: A list of unique N-Grams.
    """
    return [item for item in ngrams if ngrams.count(item) == 1]

# Function to extract shared N-Grams between two lists
def sharedNgrams(ng1, ng2):
    """
    Finds shared N-Grams between two lists.
    
    Args:
        ng1 (list): First list of N-Grams.
        ng2 (list): Second list of N-Grams.
        
    Returns:
        list: A list of shared N-Grams.
    """
    return list(set(ng1) & set(ng2))

# Function to calculate N-Grams similarity
def nGramsSimilarity(ng1, ng2):
    """
    Calculates similarity between two lists of N-Grams.
    
    Args:
        ng1 (list): First list of N-Grams.
        ng2 (list): Second list of N-Grams.
        
    Returns:
        float: Similarity score between 0 and 1.
    """
    ung1 = uniqueNgrams(ng1)
    ung2 = uniqueNgrams(ng2)
    A = len(ung1)
    B = len(ung2)
    C = len(sharedNgrams(ung1, ung2))
    return (2 * C) / (A + B)

# Calculate similarity between the first two sentences
similarity = nGramsSimilarity(extrair_ngrams_chars(frases[0], 2), extrair_ngrams_chars(frases[1], 2))
print('\nSimilarity between sentences 1 and 2: {:.2f}%'.format(similarity * 100))

# Print unique N-Grams from the first sentence
print('\nUnique N-Grams in sentence 1: {}'.format(uniqueNgrams(frases[0])))

# Word frequency analysis
wordfreq = {}
for sentence in corpus:
    tokens = word_tokenize(sentence, language='portuguese')
    for token in tokens:
        wordfreq[token] = wordfreq.get(token, 0) + 1

# Prepare frequency data for plotting
frequencies = list(wordfreq.values())
frequencies = np.array(frequencies)

# Plot frequency distribution
a = 2
count, bins, ignored = plt.hist(frequencies[frequencies < 50], 50, density=True)
x = np.arange(1., 50.)
y = x**(-a) / special.zetac(a)

plt.plot(x, y / max(y), linewidth=2, color='r')
plt.show()
