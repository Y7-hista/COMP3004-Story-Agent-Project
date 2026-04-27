from collections import defaultdict, Counter

class RNNModel:
    def __init__(self, n=3):
        self.n = n
        self.model = defaultdict(list)