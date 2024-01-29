import random
from collections import Counter

class UnigramModel:
    def __init__(self, smoothing=True):
        self.word_counts = Counter()
        self.total_words = 0
        self.vocab_size = 0
        self.smoothing = smoothing

    def train(self, corpus):
        # Count the frequency of each word in the corpus
        self.word_counts = Counter(corpus)
        self.total_words = len(corpus)
        self.vocab_size = len(set(corpus))

    def generate_sentence(self, length=10):
        # Generate a random sentence based on the unigram model
        sentence = [random.choice(list(self.word_counts.keys())) for _ in range(length)]
        return ' '.join(sentence)

    def probability(self, word):
        # Calculate the probability of a word based on the unigram model with smoothing
        if self.total_words == 0:
            return 0

        if self.smoothing:
            return (self.word_counts[word] + 1) / (self.total_words + self.vocab_size)
        else:
            return self.word_counts[word] / self.total_words

    def perplexity(self, sentence):
        # Calculate perplexity of a sentence
        words = sentence.split()
        word_probabilities = [self.probability(word) for word in words]
        entropy = -sum([prob * (1 / prob) for prob in word_probabilities]) / len(words)
        perplexity = 2 ** entropy
        return perplexity

# Example usage
corpus = ["I", "love", "natural", "language", "processing", "and", "unigram", "models"]
model = UnigramModel(smoothing=True)
model.train(corpus)

# Generate 5 sentences and calculate perplexity
for _ in range(5):
    generated_sentence = model.generate_sentence()
    perplexity = model.perplexity(generated_sentence)
    print(f"Generated Sentence: {generated_sentence}, Perplexity: {perplexity:.2f}")
