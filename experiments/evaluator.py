import re
import math
import numpy as np

from collections import Counter
from itertools import combinations
import os
import spacy
os.environ["TRANSFORMERS_NO_TF"] = "1"
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

# semantic embedding model
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

from experiments.visualization import *

class StoryEvaluator:

    def __init__(self):
        # lightweight unsafe words
        self.toxic_words = {"kill", "murder", "hate", "stupid", "idiot", "blood", "dead", "die"}

    def tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def ngrams(self, words, n):
        if len(words) < n:
            return []

        return [
            tuple(words[i:i+n])
            for i in range(len(words)-n+1)
        ]

    def split_sentences(self, story):
        return [
            s.strip()
            for s in re.split(r"[.!?]", story)
            if s.strip()
        ]

    # Keyword coverage
    def keyword_coverage(self, story, keywords):
        story = story.lower()
        hit = 0

        for k in keywords:
            if k.lower() in story:
                hit += 1

        return hit / max(1, len(keywords))

    def keyword_dispersion(self, story, keywords):
        words = self.tokenize(story)
        positions = []

        for k in keywords:
            pos = [i for i, w in enumerate(words) if w == k.lower()]

            if len(pos) > 0:
                positions.append(pos[0])

        if len(positions) < 2:
            return 0

        spread = max(positions) - min(positions)

        return spread / max(1, len(words))

    # Diversity 
    def distinct1(self, story):
        words = self.tokenize(story)

        if len(words) == 0:
            return 0

        return len(set(words)) / len(words)

    def distinct2(self, story):
        words = self.tokenize(story)
        grams = self.ngrams(words, 2)

        if len(grams) == 0:
            return 0

        return len(set(grams)) / len(grams)

    def lexical_entropy(self, story):
        words = self.tokenize(story)

        if len(words) == 0:
            return 0

        counts = Counter(words)
        total = len(words)
        entropy = 0

        for c in counts.values():
            p = c / total
            entropy -= p * math.log(p + 1e-12, 2)

        return entropy

    # Repetition rate
    def repetition_rate(self, story):
        words = self.tokenize(story)
        grams = self.ngrams(words, 3)

        if len(grams) == 0:
            return 0

        repeats = (len(grams) - len(set(grams)))

        return repeats / len(grams)

    def self_bleu_like(self, stories):
        if len(stories) < 2:
            return 0

        overlaps = []

        for a, b in combinations(stories, 2):
            A = set(self.tokenize(a))
            B = set(self.tokenize(b))
            j = len(A & B) / max(1, len(A | B))

            overlaps.append(j)

        return sum(overlaps) / len(overlaps)

    # Gramma
    def syntactic_wellformedness(self, story):
        sentences = self.split_sentences(story)

        if not sentences:
            return 0

        valid = 0

        for s in sentences:
            doc = nlp(s)

            has_subject = False
            has_verb = False

            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    has_subject = True

                if token.pos_ == "VERB":
                    has_verb = True

            if has_subject and has_verb:
                valid += 1

        return valid / len(sentences)

    # Semantic coherence
    def semantic_coherence(self, story):
        sentences = self.split_sentences(story)

        if len(sentences) < 2:
            return 0

        embeddings = semantic_model.encode(sentences)
        sims = []

        for i in range(len(embeddings)-1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            sims.append(sim)

        return float(np.mean(sims))

    # Toxicity
    def toxicity_score(self, story):
        words = self.tokenize(story)
        toxic_hits = 0

        for w in words:
            if w in self.toxic_words:
                toxic_hits += 1

        return toxic_hits / max(1, len(words))

    # Story length
    def avg_sentence_length(self, story):
        sents = self.split_sentences(story)

        if len(sents) == 0:
            return 0

        total_words = sum(len(self.tokenize(s)) for s in sents)

        return total_words / len(sents)

    # Evaluation result return
    def evaluate_runs(self, stories, keywords):
        metrics = {
            "keyword_coverage": [],
            "keyword_dispersion": [],
            "distinct_1": [],
            "distinct_2": [],
            "lexical_entropy": [],
            "avg_sentence_length": [],
            "repetition_rate": [],
            "syntax_validity": [],
            "semantic_coherence": [],
            "toxicity_score": []
        }

        for s in stories:
            metrics["keyword_coverage"].append(self.keyword_coverage(s, keywords))
            metrics["keyword_dispersion"].append(self.keyword_dispersion(s, keywords))
            metrics["distinct_1"].append(self.distinct1(s))
            metrics["distinct_2"].append(self.distinct2(s))
            metrics["lexical_entropy"].append(self.lexical_entropy(s))
            metrics["avg_sentence_length"].append(self.avg_sentence_length(s))
            metrics["repetition_rate"].append(self.repetition_rate(s))
            metrics["syntax_validity"].append(self.syntactic_wellformedness(s))
            metrics["semantic_coherence"].append(self.semantic_coherence(s))
            metrics["toxicity_score"].append(self.toxicity_score(s))

        results = {}

        for k, vals in metrics.items():
            results[k] = round(float(np.mean(vals)), 3)
            results[k + "_std"] = round(float(np.std(vals)), 3)

        results["self_bleu_like"] = round(self.self_bleu_like(stories), 3)

        return results