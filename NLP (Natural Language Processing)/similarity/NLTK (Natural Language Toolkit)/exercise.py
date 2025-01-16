from collections import OrderedDict

# Word lexicon
# List of predefined words for comparison
lexicon = ["abacate", "abacaxi", "abobora", "abobrinha", "ananás", "maça", "mamão", "manga", "melancia", "melão", "mexerica", "morango"]

# Take user input and convert it to lowercase
input_string = input("Enter a word: ")
input_string = input_string.lower()

# Define the size of n-grams
n = 2

# Generate n-grams from the input string
ngram_input = []
for i in range(len(input_string) - (n - 1)):
    gram = input_string[i:i + n]
    ngram_input.append(gram)

# Remove duplicates from the n-gram list and calculate its size
unique_input = list(OrderedDict.fromkeys(ngram_input))
A = len(unique_input)

# Display the processed input and its n-grams
print("Input: {}".format(input_string))
print("Input n-gram: {}".format(unique_input))
print("A: {}".format(A))
print("")

# Initialize similarity list
s_list = []

# Compare the input with each word in the lexicon
for j in range(len(lexicon)):
    lexicon_word = lexicon[j]
    
    # Generate n-grams for the current lexicon word
    ngram_lexicon = []
    for i in range(len(lexicon_word) - (n - 1)):
        gram = lexicon_word[i:i + n]
        ngram_lexicon.append(gram)
    
    # Remove duplicates from the lexicon n-grams
    unique_lexicon = list(OrderedDict.fromkeys(ngram_lexicon))
    B = len(unique_lexicon)
    
    # Find the intersection of n-grams between input and lexicon word
    intersection = set(unique_input).intersection(unique_lexicon)
    C = len(intersection)
    
    # Display the n-grams and counts for the current lexicon word
    print("{} B: {} C: {}".format(unique_lexicon, B, C))
    
    # Calculate similarity score S
    S = 2 * C / (A + B)
    s_list.append(S)
    print("S = {}".format(round(S, 2)))
    print("")

# Find the lexicon word with the highest similarity score
max_similarity = max(s_list)
index = s_list.index(max_similarity)

# Display the word with the highest similarity score, if significant
if max_similarity > 0.65:
    print("Highest similarity found: {} with {}%".format(lexicon[index], round(s_list[index] * 100, 2)))
else:
    print("No word with significant similarity was found in the lexicon.")

print("")
