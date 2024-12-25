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

print("Bigram probabilities matrix: ", bigram_probabilities)