# pip install gensim
# pip install nltk

# In case of an error during the installation of nltk, execute these lines in Python:
#    import nltk
#    nltk.download()
# Navigate to Models, select punkt, and click download.

####################################################################################
# Tokenization of Words
# The word_tokenize() function splits sentences into words
from nltk.tokenize import word_tokenize 

data = "Mars is approximately half the diameter of Earth."
print(word_tokenize(data))
# Output: ['Mars', 'is', 'approximately', 'half', 'the', 'diameter', 'of', 'Earth']


####################################################################################
# Tokenization of Sentences
# The sent_tokenize function splits text into sentences.
# Useful for counting the number of lines per sentence.
from nltk.tokenize import sent_tokenize

data = "Mars is a cold desert world. It is half the size of Earth. "
print(sent_tokenize(data))
# Output: ['Mars is a cold desert world', 'It is half the size of Earth ']


####################################################################################
# Open files and tokenize sentences
from nltk.tokenize import word_tokenize, sent_tokenize

file_docs = []

filePath = r"text.txt"

with open (filePath) as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file_docs.append(line)

print("Number of documents:", len(file_docs))
print(file_docs)

####################################################################################
# Tokenize Words and create a Dictionary

gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in file_docs]

# Output: [['mars', 'is', 'a', 'cold', 'desert', 'world', '.'],
#          ['it', 'is', 'half', 'the', 'size', 'of', 'earth', '.']]
import gensim

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary.token2id)
# Output: {'.': 0, 'a': 1, 'cold': 2, 'desert': 3, 'is': 4, 'mars': 5,
#          'world': 6, 'earth': 7, 'half': 8, 'it': 9, 'of': 10, 'size': 11, 'the': 12}