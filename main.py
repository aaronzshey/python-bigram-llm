# https://dev.to/santhoshvijayabaskar/build-your-own-language-model-a-simple-guide-with-python-and-numpy-1k3l
import numpy as np

corpus = """ Good heavens! Lane! Why are there no cucumber sandwiches I ordered them specially

There were no cucumbers in the market this morning, sir I went down twice

No cucumbers

No sir Not even for ready money"""
# split the words
words = corpus.lower().split()

vocab = list(set(words))
vocab_size = len(vocab)

# convert the list of words into a dictionary of words and indices

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
# enumuate turns the list of words into a list of tuples with the index and word
# word:idx is syntax that creates a key value pair in the dictionary
# for idx, word in ... is a for loop that goes through the list of tuples and creates a dictionary

# convert the dictionary of words and indices into a dictionary of indices and words
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# this is basically the opposite of above

# convert the list of words into a list of indices
corpus_indices = [word_to_idx[word] for word in words]

# init a matrix of zeros
bigram_counts = np.zeros((vocab_size, vocab_size))

# count the bigrams
# we are now populating the matrix with 1s - the row number is the current word index,
# the column number is the next word index.  this is the bigram
for i in range(len(corpus_indices) - 1):
    current_word = corpus_indices[i]
    next_word = corpus_indices[i + 1]
    bigram_counts[current_word, next_word] += 1


# Apply Laplace smoothing by adding 1 to all bigram counts
bigram_counts += 0.01

# Normalize the counts to get probabilities
bigram_probabilities = bigram_counts / bigram_counts.sum(axis=1, keepdims=True)

def predict_next_word(current_word, bigram_probabilities):
    word_idx = word_to_idx[current_word]
    next_word_probs = bigram_probabilities[word_idx]
    # randomly pick one
    next_word_idx = np.random.choice(vocab_size, p=next_word_probs)
    return idx_to_word[next_word_idx]

print("Bigram probabilities matrix: ", bigram_probabilities)

# Test the model with a word
current_word = "cucumber"
next_word = predict_next_word(current_word, bigram_probabilities)
print(f"Given '{current_word}', the model predicts '{next_word}'.")
# Given 'cucumber', the model predicts 'sandwiches'.

def generate_sentence(start_word, bigram_probabilities, length=5):
    sentence = [start_word]
    current_word = start_word

    for _ in range(length):
        next_word = predict_next_word(current_word, bigram_probabilities)
        sentence.append(next_word)
        current_word = next_word

    return ' '.join(sentence)

generated_sentence = generate_sentence("cucumber", bigram_probabilities, length=10)
print(f"Generated sentence: {generated_sentence}")
# Generated sentence: cucumber sandwiches i went down twice no cucumbers in the market