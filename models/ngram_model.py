from collections import defaultdict, Counter
import random


class NGramModel:

    def __init__(self, n=3, smoothing=1):
        self.n = n
        self.counts = defaultdict(Counter)   # context -> {word: count}
        self.context_totals = defaultdict(int)
        self.vocab = set()
        self.smoothing = smoothing  # Laplace smoothing (k=1)

    def train(self, tokens):
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i+self.n-1])
            word = tokens[i+self.n-1]

            self.counts[context][word] += 1
            self.context_totals[context] += 1
            self.vocab.add(word)

    def get_prob(self, context, word):
        """
        计算概率 P(word | context)
        使用 Laplace smoothing
        """
        context = tuple(context[-(self.n-1):])

        count = self.counts[context][word]
        total = self.context_totals[context]

        V = len(self.vocab)

        # Laplace smoothing
        prob = (count + self.smoothing) / (total + self.smoothing * V)

        return prob

    def predict(self, context):
        """
        按概率采样
        """
        context = tuple(context[-(self.n-1):])

        if context not in self.counts:
            return None

        words = list(self.vocab)
        probs = [self.get_prob(context, w) for w in words]

        return random.choices(words, weights=probs)[0]