# The minimum distance between two words is the minimum number of character 
# edits (insertions, deletions, or substitutions) needed to transform 
# one word into the other.

import nltk

word1 = 'padeiro'
word2 = 'pandeiro'
word3 = 'bombeiro'
word4 = 'padaria'

print(nltk.edit_distance(word1, word2))
print(nltk.edit_distance(word1, word3))
print(nltk.edit_distance(word1, word4))
print(nltk.edit_distance(word3, word4))
