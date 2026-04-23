from collections import defaultdict
import random

class NGramModel:

    def __init__(self, n=3):
        self.n = n
        self.model = defaultdict(list)

    def train(self, tokens):
        """
        正确逻辑：
        用前 n-1 个词预测第 n 个词
        """
        for i in range(len(tokens) - self.n + 1):
            key = tuple(tokens[i:i+self.n-1])   
            next_word = tokens[i+self.n-1]     
            self.model[key].append(next_word)

    def predict(self, context):
        """
        context: list of previous words
        """
        key = tuple(context[-(self.n-1):])

        if key in self.model:
            return random.choice(self.model[key])
        else:
            return None