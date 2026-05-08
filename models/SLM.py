# Statistical language models

import nltk
import random
import string

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.lm import MLE
from nltk.lm import Laplace, KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from collections import Counter
import pickle
import os

from models.ngram_model import NGramModel
from models.model_utils import save_model, load_model
from models.topic_planner import TopicPlanner

# Run for the first time:
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

class StatisticalLanguageModel:
    def __init__(self, n = 4, smoothing = "laplace", planner = None):
        self.n = n
        self.smoothing = smoothing
        self.model = None
        self.planner = planner
        self.unigrams = []
        self.bigrams = []
        self.trigrams = []
        self.fourgrams = []
        self.tokenized_text = []
        

    def preprocess(self, text):
        print("Preprocessing")
        punctuation = string.punctuation + "'" + "-" + "'" + "-"
        punctuation = punctuation.replace(".", "")
        text = text.lower()

        for p in punctuation:
            if p not in ".?!":
                text = text.replace(p, " ")
        sentences = sent_tokenize(text)
        processed = []

        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            tokens = [
                t for t in tokens
                    if t not in [".", "!", "?"]
            ]
            if len(tokens) > 0:
                processed.append(tokens)
        return processed
    
    def build_grams(self):
        for seq in self.tokenized_text:
            self.unigrams.extend(seq)
            self.bigrams.extend(list(ngrams(seq, 2)))
            self.trigrams.extend(list(ngrams(seq, 3)))
            self.fourgrams.extend(list(ngrams(seq, 4)))
        freq_uni = nltk.FreqDist(self.unigrams)
        freq_bi = nltk.FreqDist(self.bigrams)
        freq_tri = nltk.FreqDist(self.trigrams)
        freq_four = nltk.FreqDist(self.fourgrams)
        print("5 most common unigrams:" + str(freq_uni.most_common(5)))
        print("5 most common bigrams: " + str(freq_bi.most_common(5)))
        print("5 most common trigrams: " + str(freq_tri.most_common(5)))
        print("5 most common fourgrams: " + str(freq_four.most_common(5)))
    
    def build_lm(self):
        print("Building Language Models.")

        train_data, vocab = padded_everygram_pipeline(self.n, self.tokenized_text)

        if self.smoothing  ==  "mle":
            self.model = MLE(self.n)
        elif self.smoothing  ==  "laplace":
            self.model = Laplace(self.n)
        # elif self.smoothing  ==  "kneserney":
        #     self.model = KneserNeyInterpolated(self.n, discount = 0.1)
        else:
            raise ValueError("unknown smoothing")
        
        self.model.fit(train_data, vocab)
        print(f"{self.smoothing} model built")

    def train(self, text, model_path = "saved_models/SLM_model.pkl"):
        if os.path.exists(model_path):
            print("Loading saved model...")
            self.model = load_model(model_path)
            return
        
        self.tokenized_text = self.preprocess(text)
        self.build_grams()
        self.build_lm()

        save_model(self.model, model_path)
        print("Model saved.")

    # LM Function
    def sentence_probability(self, sentences):
        tokens = word_tokenize(sentences.lower())
        probability = 1
        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - (self.n - 1) : i])
            word = tokens[i]
            p = self.model.score(word, context)
            probability *=  p
        return probability

    # Test model perplexity
    def perplexity(self, test_sentences):
        test = [word_tokenize(test_sentences.lower())]
        test_grams, _ = padded_everygram_pipeline(self.n, test)
        test_grams = list(test_grams)[0]

        return self.model.perplexity(test_grams)

    # next word prediction
    def next_word(self, context):
        context = tuple(word_tokenize(context.lower())[-(self.n - 1) : ])

        vocab = list(self.model.vocab)
        scored = []

        for w in vocab:
            p = self.model.score(w, list(context))
            scored.append((w, p))

        scored = sorted(scored, key = lambda x:x[1], reverse = True)

        return scored[:10]

    # Generating Language
    def generate(self, keywords, num_sentences = 8, max_sentence_len = 18):
        """
        Topic conditioned decoding
        with entropy-adaptive sampling
        """
        if self.planner is None:
            raise ValueError("TopicPlanner not injected into StatisticalLanguageModel")

        plan = self.planner.build_topic_plan(keywords)

        seed = plan["seed"]
        topic_plan = plan["plan"]
        neighbors = plan["neighbors"]

        context = list(seed[-(self.n-1):])
        story = []
        unused = set(k.lower() for k in keywords)

        for s in range(num_sentences):
            sentence = []

            if s < len(topic_plan):
                target = topic_plan[s]
            else:
                target = random.choice(keywords).lower()

            for t in range(max_sentence_len):
                candidates = []
                for w in self.model.vocab:
                    if w in ["<s>","</s>"]:
                        continue

                    p = self.model.score(w, context)
                    if p <= 0:
                        continue

                    # adaptive temperature
                    if self.n == 2:
                        temp = 1.15
                    else:
                        temp = 1.65

                    p = p**(1/temp)

                    # SOFT topic bias
                    # much weaker
                    if w == target:
                        p *= 1.8

                    if (target in neighbors and w in neighbors[target]):
                        p*= 1.15

                    if w in unused:
                        p *= 1.15

                    # anti repetition
                    if w in sentence[-4:]:
                        p *= 0.05

                    candidates.append((w,p))

                if not candidates:
                    break

                # true top-p (correct)
                candidates.sort(key = lambda x:x[1], reverse = True)
                total = sum(p for _, p in candidates)

                norm = [(w,p/total) for w,p in candidates]

                # entropy adaptive p
                if self.n == 2:
                    nucleus_p = 0.93
                else:
                    nucleus_p = 0.97

                filtered = []
                cum = 0

                for w,p in norm:
                    filtered.append((w,p))
                    cum += p

                    if cum >= nucleus_p:
                        break

                words = [x[0] for x in filtered]
                probs = [x[1] for x in filtered]

                next_word = random.choices(words, weights = probs, k = 1)[0]
                sentence.append(next_word)
                context.append(next_word)
                context = context[-(self.n-1):]

                if next_word.lower() in unused:
                    unused.remove(next_word.lower())

                if (len(sentence)>10 and random.random()<0.22):
                    break

            if len(sentence)>5:
                txt = " ".join(sentence)
                txt = txt.capitalize()+"."
                story.append(txt)

        # light repair only
        text = " ".join(story).lower()

        missing = [k for k in keywords if k.lower() not in text]

        # if missing topics absent,
        # regenerate last sentence under lexical bias
        if len(missing) > 0:
            repair_sentence = []
            target = random.choice(missing)

            for i in range(max_sentence_len):
                candidates = []

                for w in self.model.vocab:
                    if w in ["<s>","</s>"]:
                        continue

                    p = self.model.score(w, context)

                    if p <= 0:
                        continue
                    # strong temporary bias only in repair decoding
                    if w == target:
                        p *= 4.5

                    # no repeat penalty
                    if w in repair_sentence[-4:]:
                        p *= 0.05

                    candidates.append((w,p))

                if not candidates:
                    break

                candidates.sort(key = lambda x:x[1], reverse = True)
                # nucleus repair decoding
                total = sum(p for j, p in candidates)
                cum = 0
                filtered = []

                for w,p in candidates:
                    p = p / total
                    filtered.append((w,p))
                    cum += p

                    if cum >= 0.92:
                        break

                words = [x[0] for x in filtered]
                probs = [x[1] for x in filtered]

                nxt = random.choices(words, weights = probs, k = 1)[0]
                repair_sentence.append(nxt)
                context.append(nxt)
                context = context[-(self.n-1):]

                if len(repair_sentence)>10 and random.random()<0.2:
                    break

            if len(repair_sentence)>4:
                story.append(" ".join(repair_sentence).capitalize() + ".")
                
        return " ".join(story)   

    # compare models
    def compare_smoothing(self, sentence):
        for m in ["mle", "laplace"]:
            tmp = StatisticalLanguageModel(n = self.n, smoothing = m)
            flat = " ".join(
                [
                 " ".join(x)
                 for x in self.tokenized_text
                ]
            )
            tmp.train(flat)

            print(m, tmp.perplexity(sentence))

